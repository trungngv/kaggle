source('src/common.R')

## Read data ----
raw <- read_json("data/train.json")
raw_test <- read_json("data/test.json")
saved_raw <- raw
saved_test <- raw_test

# this is for document vector, but it's better to use features directly
raw <- mutate(raw, n_features = sapply(features, length))
raw_test <- mutate(raw_test, n_features = sapply(features, length))
raw$features <- sapply(raw$features, function(x) {paste0(gsub(" ", "_", x), collapse = " ")}) %>% as.vector()
raw_test$features <- sapply(raw_test$features, function(x) {paste0(gsub(" ", "_", x), collapse = " ")}) %>% as.vector()
# purrr style
#v_features <- map_at(raw$features, 1:nrow(raw), ~ paste0(unlist(.x), collapse = " ")) %>% unlist() %>% as.vector()

## Data pre-processing -----
m <- rbind(select(raw, -interest_level), raw_test)
m <- m %>% add_date_time() %>% clean_with_building_id() %>% clean_display() %>% 
  clean_street() %>% add_latlong() %>% add_bed_bath()
cat_vars <- c('manager_id', 'building_id', 'display_address', 'street_address', 'latlong',
              'build_bed', 'display_bed', 'street_bed', 'latlong_bed')
features <- cv_target_encoding(data.frame(m[1:nrow(raw),], interest_level=raw$interest_level),
                               m[-(1:nrow(raw)),], cat_vars[1:2], 10)
train_features <- left_join(select(raw, listing_id), features$train)
test_features <- left_join(select(raw_test, listing_id), features$test)
m <- inner_join(m, rbind(train_features, test_features))

#build_bed_n is changed due to this grouping
# Convert categorical variables to integer (for xgboost),
# grouping all levels with only 1 observation as 1
for (v in cat_vars) {
  cnts <- m %>% group_by_(v) %>% summarise(n = n()) %>% filter(n <= 10)
  colnames(cnts) <- c("V1", "n")
  m[[v]] <- as.integer(as.factor(ifelse(m[[v]] %in% cnts$V1, "-1", m[[v]])))
}
m <- summary_stats(m, cat_vars)

# some other features
m <- mutate(m, n_photos = sapply(photos, length))

# This feature compares the predicted price 
# and the actual price (to estimate over/under pricing)
# It uses all of data including test to build such as model
price_regressor <- function(m) {
  # build a very simple pricing model, like what a human would do
  m <- select(m, bathrooms, bedrooms, latitude, longitude, n_photos, n_features, price) %>% filter(price < 100000)
  ctrl <- trainControl(method = "cv", number = 2, search = "random", allowParallel = TRUE)
  model <- train(price ~ ., data = m, method = "xgbTree", trControl = ctrl, tuneLength=5)
  model
}
price_model <- price_regressor(m)
m <- m %>% mutate(predicted_price = predict(price_model, m),
                 ratio_predicted_price = predicted_price / price)
m <- preprocess_numeric(m, na = 0)
write_csv(m, 'data/processed_numeric.csv')

# Add text features ----
corpus <- Corpus(DataframeSource(rbind(select(raw, features), select(raw_test, features))))
tdm <- DocumentTermMatrix(corpus, list(removePunctuation = TRUE, stopwords = TRUE, stemming = FALSE,
                                       weighting=weightBin, removeNumbers = TRUE,
                                       bounds=list(global=c(10, Inf))))
d_txt <- as.data.frame(as.matrix(tdm))
write_csv(d_txt, 'data/text.csv')

m <- cbind(d_txt, m)

## Save processed data ----
d <- cbind(m[1:nrow(raw),], interest_level = as.character(raw$interest_level))
d_test <- m[(nrow(raw)+1):nrow(m),]
write_csv(d, 'data/mine_train.csv')
write_csv(d_test, 'data/mine_test.csv')
