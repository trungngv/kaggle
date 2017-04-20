library(data.table)
library(plyr); library(dplyr)
library(jsonlite)
library(lubridate)
library(ggplot2)
library(corrplot)
library(plotly)
library(readr)
library(stringr)
library(doParallel)
library(dummies)
library(xgboost)
library(tidyr)
library(tm)
library(caret)
packages <- c("jsonlite", "dplyr", "purrr")
purrr::walk(packages, library, character.only = TRUE, warn.conflicts = FALSE)

# Read raw data from json file
read_json <- function(json_file) {
  raw <- fromJSON(json_file)
  vars <- setdiff(names(raw), c("photos", "features"))
  raw <- map_at(raw, vars, unlist) %>% tibble::as_tibble(.)
  raw$display_address <- tolower(str_trim(raw$display_address))
  raw$street_address <- tolower(str_trim(raw$street_address))
  raw
}

add_date_time <- function(m) {
  dt_created <- as_datetime(m$created)
  m <- mutate(m, month_created = month(dt_created),
              wday_created = wday(dt_created),
              hour_created = hour(dt_created))
  # there are multiple listings for a property
  m <- group_by(m, manager_id, description) %>% 
    mutate(next_post = lag(created, order_by=desc(created))) %>% 
    mutate(duration = as.numeric(difftime(ifelse(is.na(next_post), '2016-07-01', next_post), created, unit='days'))) %>%
    select(-next_post) %>% ungroup()
  m
}

add_latlong <- function(d) {
  disc_lat <- as.integer(cut(d$latitude, c(-Inf, seq(40.6, 40.9, length.out=10), Inf), labels=FALSE))
  disc_long <- as.integer(cut(d$longitude, c(-Inf, seq(-74.05, -73.8, length.out=10), Inf), labels=FALSE))
  d$latlong <- as.integer(as.factor(sprintf('%s_%s', disc_lat, disc_long)))
  d$lat_price <- d$price / (1 + d$latitude)
  d$long_price <- d$price / (1 + d$longitude)
  d
}

add_bed_bath <- function(d) {
  mutate(d,
         bed_bath = paste(bedrooms, bathrooms, sep="_"),
         build_bed = paste(building_id, bedrooms, sep="_"),
         display_bed = paste(display_address, bedrooms, sep="_"),
         street_bed = paste(street_address, bedrooms, sep="_"),
         latlong_bed = paste(latlong, bedrooms, sep="_")
         )
}

summary_stats_bed <- function(g, colname_prefix) {
  result <- summarise(g, bed1=mean(bedrooms))
  colnames(result)[-1] <- paste0(colname_prefix, '_', names(result)[-1])
  result
}

summary_stats_price <- function(g, colname_prefix) {
  result <- summarise(g, pr1=mean(price))
  colnames(result)[-1] <- paste0(colname_prefix, '_', names(result)[-1])
  result
}

# compute stats for each categorical variable
# use describe: n, mean, med, std
summary_stats <- function(m, cat_vars) {
  stats <- list()
  for (i in 1:length(cat_vars)) {
    grouped <- m %>% group_by_(cat_vars[i])
    cnts <- summarise(grouped, n = n())
    colnames(cnts)[2] <- sprintf('%s_n', cat_vars[i])
    beds <- summary_stats_bed(grouped, cat_vars[i])
    prices <- summary_stats_price(grouped, cat_vars[i])
    #baths <- summary_stats_bath(grouped, cat_vars[i])
    #stats[[i]] <- inner_join(cnts, beds) %>% inner_join(prices) %>% inner_join(baths)
    stats[[i]] <- inner_join(cnts, prices) %>% inner_join(beds)
  }
  result <- m
  for (i in 1:length(stats)) {
    result <- inner_join(result, stats[[i]])
  }
  result
}

# This does the actual encoding...
encode_categorical_var <- function(d, var, prefix, limit = 10) {
  # only keep the category with counts exceeding the lower limit
  # filter by total or by count for each type??
  interest_level <- 'interest_level'
  result <- d %>% group_by_(var, interest_level) %>%
    summarise(n = n()) %>% 
    # this gives count by var and interest_level
    spread(key = interest_level, value = n, fill = 0) %>%
    filter(high + low + medium >= limit) %>%
    # computes proportions
    transmute(prop_high = high / (high + low + medium),
           prop_med = medium / (high + low + medium))
  colnames(result) <- paste0(prefix, "_", names(result))
  colnames(result)[1] <- var
  result
} 

# Encode targets for high cardinal categorical variables.
# This uses out of sample, i.e. the targets of 4 folds are encoded and
# used for the remaining fold. This is similar to the procedure to build
# meta features for stacking, but model is the encoding algorithm.
# Also encode for d_test, which is average of the predictions by 5 folds.
cv_target_encoding <- function(d, d_test, cat_vars, limit) {
  set.seed(0404)
  nfolds <- 5
  folds <- createFolds(d$interest_level, 5)
  new_train <- data.frame()
  new_test <- data.frame()
  for (i in 1:length(folds)) {
    # these are used for 'training' / encoding
    in_samples <- d[-folds[[i]],]
    out_samples <- d[folds[[i]],]
    res <- list()
    for (i in 1:length(cat_vars)) {
      res[[i]] <- encode_categorical_var(in_samples, cat_vars[i], cat_vars[i], limit)
    }
    tmp <- out_samples
    tmp2 <- d_test
    for (i in 1:length(res)) {
      tmp <- left_join(tmp, res[[i]])
      tmp2 <- left_join(tmp2, res[[i]])
    }
    new_train <- bind_rows(new_train, tmp)
    new_test <- bind_rows(new_test, tmp2)
  }
  new_test <- select(new_test, matches(".*_prop_.*"), listing_id) %>%
    # average encoding by all 5 folds
    group_by(listing_id) %>% summarise_all(funs(mean(., na.rm=TRUE)))
  result = list(train = select(new_train, matches(".*_prop_.*"), listing_id),
                test = new_test)
}

clean_with_building_id_ <- function(m, var) {
  bld_id <- 'building_id'
  cnts <- m %>% group_by_(bld_id, var) %>% summarise(n = n()) %>% 
    group_by_(bld_id) %>% top_n(1) %>% dplyr::slice(1) %>% filter(building_id != 0) %>% select(-n)
  colnames(cnts) <- c('building_id', 'new')
  m <- left_join(m, cnts)
  m[[var]] <- ifelse(m$building_id != 0, m$new, m[[var]])
  m$new <- NULL
  m
}

# Assuming same building id should have same display address, lat, and long,
# use the most common values given building id to get a bit more clean data
# Except building_id = 0
clean_with_building_id <- function(m) {
  # display address
  m %>% clean_with_building_id_('display_address') %>% 
    clean_with_building_id_('latitude') %>%
    clean_with_building_id_('longitude')
}

clean_street <- function(m) {
  m <- m %>%
    mutate(street_address = str_replace_all(street_address, '\\.|,', ' ')) %>%
    mutate(street_address = str_trim(street_address)) %>%
    mutate(street_address = str_replace(street_address, " ave$", " avenue")) %>%
    mutate(street_address = str_replace(street_address, " st$", " street")) %>%
    mutate(street_address = str_replace(street_address, " blvd$", " boulevard")) %>%
    mutate(street_address = str_replace(street_address, " pl$", " place")) %>%
    mutate(street_address = str_replace(street_address, "(^e | e )", " east ")) %>%
    mutate(street_address = str_replace(street_address, "(^w | w )", " west ")) %>%
    mutate(street_address = str_replace(street_address, "(^n | n )", " north ")) %>%
    mutate(street_address = str_replace(street_address, "(^s | s )", " south ")) %>%
    mutate(street_address = str_replace(street_address, "([0-9]+)(st|th|nd) ", "\\1 "))
}

clean_display <- function(m) {
  m <- m %>%
    mutate(display_address = str_replace_all(display_address, '\\.|,', ' ')) %>%
    mutate(display_address = str_trim(display_address)) %>%
    mutate(display_address = str_replace(display_address, " ave$", " avenue")) %>%
    mutate(display_address = str_replace(display_address, " st$", " street")) %>%
    mutate(display_address = str_replace(display_address, " blvd$", " boulevard")) %>%
    mutate(display_address = str_replace(display_address, " pl$", " place")) %>%
    mutate(display_address = str_replace(display_address, "(^e | e )", " east ")) %>%
    mutate(display_address = str_replace(display_address, "(^w | w )", " west ")) %>%
    mutate(display_address = str_replace(display_address, "(^n | n )", " north ")) %>%
    mutate(display_address = str_replace(display_address, "(^s | s )", " south ")) %>%
    mutate(display_address = str_replace(display_address, "([0-9]+)(st|th|nd) ", "\\1 "))
}

# Actual pre-process of data, using manager_features and street_features from training set
# Keep numeric features only
preprocess_numeric <- function(m, na = NULL) {
  # some ratio features (to estimate expensiveness say of price per bedroom)
  m <- mutate(m,
              diff_bedbath = bedrooms - bathrooms,
              sum_bedbath = bedrooms + bathrooms,
              ratio_price_bed = price / (1 + bedrooms),
              ratio_price_bath = price / (1 + bathrooms),
              ratio_price_bedbath = price / (bedrooms + bathrooms),
              ratio_bed_bath = bedrooms / (1 + bathrooms),
              ratio_price_street1 = price / street_address_pr1,
              ratio_price_street2 = price / street_bed_pr1,
              ratio_price_bld1 = price / building_id_pr1,
              ratio_price_bld2 = price / build_bed_pr1,
              ratio_price_disp1 = price / display_address_pr1,
              ratio_price_disp2 = price / display_bed_pr1,
              ratio_price_latlong1 = price / latlong_pr1,
              ratio_price_latlong2 = price / latlong_bed_pr1
)

  # set NA to some value
  if (!is.null(na)) {
    m[is.na(m)] <- na
  }
  numeric_features <- names(m)[which(sapply(m, is.numeric) | sapply(m, is.integer))]
  dplyr::select(m, one_of(numeric_features))
}

unload <- function() {
  unloadNamespace("caret")
  unloadNamespace("ggplot2")
  unloadNamespace("reshape2")
  unloadNamespace("scales")
  unloadNamespace("plyr")
}

to_numeric <- function(label) {
  ifelse(label == 'high', 2, ifelse(label == 'medium', 1, 0))
}

probs_to_label <- function(probs) {
  probs <- mutate(probs, label = ifelse(high > medium, ifelse(high > low, 'high', 'low'), ifelse(medium > low, 'medium', 'low')))
}

# Evaluate performance (log-loss)
logloss <- function(target, pred_prob) {
  pred_prob[pred_prob == 0] <- 1e-5
  target_df <- dummy(target)
  -(sum(log(pred_prob) * target_df) / nrow(target_df))
}

case_weights <- function(interest_level) {
  w_high <- sum(interest_level == 'low') / sum(interest_level == 'high')
  w_medium <- sum(interest_level == 'low') / sum(interest_level == 'medium')
  w_low <- 1
  w_high <- 1.2
  w_medium <- 1
  weights <- ifelse(interest_level == 'high', w_high, ifelse(interest_level == 'medium', w_medium, w_low))
}

# convert multiclass labels to binary (for caret)
multiclass_to_binary <- function(multiclass_label, clas) {
  # new label = 0 for given clas, 1 for others
  if (clas == 'low') { # low vs. all
    new_labels <- ifelse(multiclass_label == 'low', 'low', 'other')
  } else if (clas == 'medium') {
    new_labels <- ifelse(multiclass_label == 'medium', 'medium', 'other')
  } else {
    new_labels <- ifelse(multiclass_label == 'high', 'high', 'other')
  }
  new_labels
}

# train binary classifier on dt and makes predictions on dv
one_vs_all_classifier <- function(dt, dv, dtest, clas, tl=5) {
  dt$interest_level <- as.factor(multiclass_to_binary(dt$interest_level, clas))
  ctrl <- trainControl(method = "cv", number = 2, search = "random", classProbs = TRUE,
                       summaryFunction = twoClassSummary, allowParallel = TRUE)
  model <- train(interest_level ~ ., data = dt, method = "xgbTree",
                 metric="ROC", trControl = ctrl, tuneLength=tl)
  list(model = model,
       val_prob = predict(model, dv, type='prob'),
       val_label = predict(model, dv),
       test_prob = predict(model, dtest, type='prob'))
}

binary_classifier <- function(dt, dv, dtest, clas1, clas2, tl=5) {
  dt <- filter(dt, interest_level == clas1 | interest_level == clas2)
  ctrl <- trainControl(method = "cv", number = 2, search = "random", classProbs = TRUE,
                       summaryFunction = twoClassSummary, allowParallel = TRUE)
  model <- train(interest_level ~ ., data = dt, method = "xgbTree",
                 metric="ROC", trControl = ctrl, tuneLength=tl)
  list(model = model,
       val_prob = predict(model, dv, type='prob'),
       val_label = predict(model, dv),
       test_prob = predict(model, dtest, type='prob'))
}

multi_classifier <- function(dt, dv, dtest, nrounds, weights=NULL) {
  registerDoParallel(4)
  getDoParWorkers()
  params <- list(eta = .03, gamma = 1, max_depth = 4, min_child_weight = 1,
                 subsample = .7, colsample_bytree = .7,
                 num_class = 3, objective = "multi:softprob", eval_metric = "mlogloss")
  if (is.null(weights)) {
    m_dt <- xgb.DMatrix(data.matrix(select(dt, -interest_level)),
                        label=to_numeric(dt$interest_level))
  } else {
    m_dt <- xgb.DMatrix(data.matrix(select(dt, -interest_level)),
                        label=to_numeric(dt$interest_level),
                        weight=weights)
  }
  m_dv <- xgb.DMatrix(data.matrix(select(dv, -interest_level)),
                             label=to_numeric(dv$interest_level))
  m_dtest <- xgb.DMatrix(data.matrix(dtest))
  # this is only for estimating error, not choosing hyperparameters
  model <- xgb.train(params, m_dt, 
                         nrounds = nrounds, nthread = 4, verbose = TRUE
                         , early.stop.round = 100
                         , watchlist = list(val = m_dv))
  val_prob <- matrix(predict(model, m_dv), ncol = 3, byrow=T) %>% data.frame()
  colnames(val_prob) <- c("p_multi_low", "p_multi_medium", "p_multi_high")
  test_prob <- matrix(predict(model, m_dtest), ncol = 3, byrow=T) %>% data.frame()
  colnames(test_prob) <- c("p_multi_low", "p_multi_medium", "p_multi_high")
  list(model = model, val_prob = val_prob, test_prob = test_prob)
}
