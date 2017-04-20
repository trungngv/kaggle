# Processed data:
# 1) processed_numeric.csv: train & test data for numeric features only
# 2) text.csv: text features from features
# 3) mine_train.csv, mine_test.csv (train + test data combined from 1 & 2, using my approach)
# 4) other_train.csv, other_test.csv (train + test data from Kaggle script)
# 5) final_train.csv, final_test.csv (other + ratio & prop features of mine)
# stacking v8 + stacknet v2 => 0.523xx

# TODO: 
# if there are building and bed / building and bed bath that have different interest levels,
# see where the difference come from?

source('src/common.R')

raw <- read_json("data/train.json")
raw_test <- read_json("data/test.json")

train1 <- fread('data/other_train.csv')
test1 <- fread('data/other_test.csv')
tt1 <- rbind(train1, test1) %>% select(listing_id, matches("feature_.*"), one_of("building_id_mean_med", "building_id_mean_high", "manager_id_mean_med", "manager_id_mean_high"))
tt2 <- fread('data/processed_numeric.csv')
tt <- inner_join(tt1, tt2, by='listing_id')
d <- cbind(tt[1:nrow(train1),], interest_level = raw$interest_level)
d$interest_level <- as.character(d$interest_level)
d_test <- tt[-(1:nrow(train1)),]

write_csv(d, 'data/final_train.csv')
write_csv(d_test, 'data/final_test.csv')

d <- fread('data/final_train.csv')
d_test <- fread('data/final_test.csv')

# meta batch 1 ----
# bunch of classifiers
# for each fold, build a model on the remaining folds, then predict
set.seed(290828)
folds <- createFolds(d$interest_level, 5)
meta_train <- data.frame()
meta_test <- data.frame()
for (i in 1:5) {
  print(sprintf('fold %d', i))
  d_train <- d[-folds[[i]],]
  d_val <- d[folds[[i]],]
  multi_clf <- multi_classifier(d_train, d_val, d_test, nrounds=1200)
  low_clf <- one_vs_all_classifier(d_train, d_val, d_test, 'low', tl=5)
  med_clf <- one_vs_all_classifier(d_train, d_val, d_test, 'medium', tl=5)
  high_clf <- one_vs_all_classifier(d_train, d_val, d_test, 'high', tl=5)
  low_med_clf <- binary_classifier(d_train, d_val, d_test, 'low', 'medium', 5)
  high_med_clf <- binary_classifier(d_train, d_val, d_test, 'high', 'medium', 5)
  high_low_clf <- binary_classifier(d_train, d_val, d_test, 'high', 'low', 5)
  meta <- data.frame(listing_id = d_val$listing_id,
                         p_low = low_clf$val_prob[, 1],
                         p_med = med_clf$val_prob[, 1],
                         p_high = high_clf$val_prob[, 1],
                         # get prob for medium as it seems like the harder case
                         p_low_med = low_med_clf$val_prob[, 2],
                         p_high_med = high_med_clf$val_prob[, 2],
                         p_high_low = high_low_clf$val_prob[, 1])
  meta <- cbind(meta, multi_clf$val_prob)
  meta_train <- rbind(meta_train, meta)
  meta <- data.frame(listing_id = d_test$listing_id,
                          p_low = low_clf$test_prob[, 1],
                          p_med = med_clf$test_prob[, 1],
                          p_high = high_clf$test_prob[, 1],
                          # get prob for medium as it seems like the harder case
                          p_low_med = low_med_clf$test_prob[, 2],
                          p_high_med = high_med_clf$test_prob[, 2],
                          p_high_low = high_low_clf$test_prob[, 1])
  meta <- cbind(meta, multi_clf$test_prob)
  meta_test <- rbind(meta_test, meta)
}
dim(meta_train)
meta_test <- meta_test %>% group_by(listing_id) %>% summarise_all(mean)
meta_train <- inner_join(meta_train, select(d, listing_id, interest_level))
write_csv(meta_train, 'data/meta_train.csv')
write_csv(meta_test, 'data/meta_test.csv')

# Meta batch 2----
# weighted classifiers
set.seed(290828)
folds <- createFolds(d$interest_level, 5)
meta_train <- data.frame()
meta_test <- data.frame()
for (i in 1:5) {
  sprintf('fold %d', i)
  d_train <- d[-folds[[i]],]
  d_val <- d[folds[[i]],]
  # weighted cases for higher recall of medium and high
  weights <- ifelse(d_train$interest_level == 'medium', 5, 1)
  biased_med <- multi_classifier(d_train, d_val, d_test, nrounds=1000, weights=weights)
  weights <- ifelse(d_train$interest_level == 'high', 10, 1)
  biased_high <- multi_classifier(d_train, d_val, d_test, nrounds=1000, weights = weights)
  meta <- data.frame(listing_id = d_val$listing_id,
                     # get prob for medium as it seems like the harder case
                     biased_med$val_prob,
                     biased_high$val_prob)
  meta_train <- rbind(meta_train, meta)
  meta <- data.frame(listing_id = d_test$listing_id,
                     biased_med$test_prob,
                     biased_high$test_prob)
  meta_test <- rbind(meta_test, meta)
}
dim(meta_train)
meta_test2 <- meta_test %>% group_by(listing_id) %>% summarise_all(mean)
meta_train2 <- inner_join(meta_train, select(d, listing_id, interest_level))
write_csv(meta_train2, 'data/meta_train2.csv')
write_csv(meta_test2, 'data/meta_test2.csv')

# meta batch 3 -----
# 'simple' model with limited features set
set.seed(290828)
folds <- createFolds(d$interest_level, 5)
meta_train <- data.frame()
meta_test <- data.frame()
few_vars <- c("listing_id", "bathrooms", "bedrooms", "price", "photo_count", "desc_wordcount")
dx <- select(d, one_of(few_vars), dplyr::contains("ratio"), interest_level)
dx_test <- select(d_test, one_of(few_vars), dplyr::contains("ratio"))
m_dtest <- xgb.DMatrix(data.matrix(dx_test))
params <- list(eta = .15, gamma = 6, max_depth = 4, min_child_weight = 5,
               subsample = .7, colsample_bytree = .5,
               num_class = 3, objective = "multi:softprob", eval_metric = "mlogloss")
for (i in 1:5) {
  d_train <- dx[-folds[[i]],]
  d_val <- dx[folds[[i]],]
  m_dt <- xgb.DMatrix(data.matrix(select(d_train, -interest_level)), label=to_numeric(d_train$interest_level))
  m_dv <- xgb.DMatrix(data.matrix(select(d_val, -interest_level)))
  model <- xgb.train(params, m_dt, nrounds = 300, nthread = 4, verbose = TRUE
                     , watchlist = list(train = m_dt))
  val_prob <- matrix(predict(model, m_dv), ncol = 3, byrow=T) %>% data.frame()
  test_prob <- matrix(predict(model, m_dtest), ncol = 3, byrow=T) %>% data.frame()
  meta <- data.frame(listing_id = d_val$listing_id, val_prob)
  meta_train <- rbind(meta_train, meta)
  meta <- data.frame(listing_id = d_test$listing_id, test_prob)
  meta_test <- rbind(meta_test, meta)
}
dim(meta_train)
meta_test3 <- meta_test %>% group_by(listing_id) %>% summarise_all(mean)
meta_train3 <- inner_join(meta_train, select(d, listing_id, interest_level))
write_csv(meta_train3, 'data/meta_train3.csv')
write_csv(meta_test3, 'data/meta_test3.csv')

predict_caret_model <- function(model, prefix, d_val, d_test) {
  val_prob <- predict(model, d_val, type='prob')
  colnames(val_prob) <- paste0(prefix, colnames(val_prob))
  print(sprintf('logloss: %.4f', logloss(d_val$interest_level, val_prob)))
  test_prob <- predict(model, d_test, type='prob')
  colnames(test_prob) <- paste0(prefix, colnames(test_prob))
  list(val_prob = val_prob, test_prob = test_prob)
}

# meta batch 4 - other models ---- 
set.seed(290828)
folds <- createFolds(d$interest_level, 5)
d <- select(d, -dplyr::contains('feature'), -manager_id, -building_id, -display_address, -street_address)
d_test <- select(d_test, -dplyr::contains('feature'), -manager_id, -building_id, -display_address, -street_address)
d[d == Inf] <- -1
d_test[d_test == Inf] <- -1
meta_train <- data.frame()
meta_test <- data.frame()
for (i in 1:5) {
  d_train <- d[-folds[[i]],]
  d_val <- d[folds[[i]],]
  # knn models
  print('knn, price, bath, bed')
  ctrl <- trainControl(method='cv', number=3)
  grid <- expand.grid(k=c(19))
  knn <- train(interest_level ~ price + bedrooms + bathrooms,
               data = d_train, method="knn",
               trControl=ctrl, preProcess = c("center", "scale"), tuneGrid=grid)
  print(knn)
  res_knn <- predict_caret_model(knn, 'knn_', d_val, d_test)

  print('knn, lat long')
  ctrl <- trainControl(method='cv', number=2)
  grid <- expand.grid(k=c(7))
  knn <- train(interest_level ~ latitude + longitude,
               data = d_train, method="knn",
               trControl=ctrl,tuneGrid=grid)
  print(knn)
  res_knn2 <- predict_caret_model(knn, 'knn2_', d_val, d_test)

  # glmnet
  #ctrl <- trainControl(method='cv', number=2)
  #grid <- expand.grid(alpha=0.325, lambda=0.0008)
  #glm <- train(interest_level ~ .,
  #             data = d_train, method="glmnet",
  #             trControl=ctrl, preProcess = c("center", "scale"), tuneGrid=grid)
  #print(glm)  
  #res_glm <- predict_caret_model(glm, 'glm_', d_val, d_test)

  # concat data
  meta <- data.frame(listing_id = d_val$listing_id, res_knn$val_prob, res_knn2$val_prob)
  meta_train <- rbind(meta_train, meta)
  meta <- data.frame(listing_id = d_test$listing_id, res_knn$test_prob, res_knn2$test_prob)
  meta_test <- rbind(meta_test, meta)
}
dim(meta_train)
meta_test4 <- meta_test %>% group_by(listing_id) %>% summarise_all(mean)
meta_train4 <- inner_join(meta_train, select(d, listing_id, interest_level))
write_csv(meta_train4, 'data/meta_train4.csv')
write_csv(meta_test4, 'data/meta_test4.csv')

# combine meta batches ----
meta_train1 <- read_csv('data/meta_train.csv')
meta_test1 <- read_csv('data/meta_test.csv')
meta_train2 <- read_csv('data/meta_train2.csv')
meta_test2 <- read_csv('data/meta_test2.csv')
meta_train3 <- read_csv('data/meta_train3.csv')
meta_test3 <- read_csv('data/meta_test3.csv')
meta_train4 <- read_csv('data/meta_train4.csv')
meta_test4 <- read_csv('data/meta_test4.csv')
meta_train <- inner_join(meta_train1, meta_train2, by=c('listing_id', 'interest_level')) %>% 
  inner_join(meta_train3, by=c('listing_id', 'interest_level')) #%>% inner_join(meta_train4, by=c('listing_id', 'interest_level'))
meta_test <- inner_join(meta_test1, meta_test2, by='listing_id') %>% 
  inner_join(meta_test3, by='listing_id') #%>% inner_join(meta_test4, by='listing_id')
meta_train[is.na(meta_train)] <- -1
meta_test[is.na(meta_test)] <- -1

# train stacker based on meta features ----
# caret, 0.5212493, nrounds ~ 500
ctrl <- trainControl(method = "cv", number = 5, search = "random", classProbs = TRUE,
                     summaryFunction = mnLogLoss, allowParallel = TRUE)
stacker <- train(interest_level ~ ., data = meta_train, method = "xgbTree",
                 metric="logLoss", trControl = ctrl, tuneLength=5)
write_rds(stacker, 'models/stacker_v10.rds')
stk_prob <- select(predict(stacker, meta_test, type='prob'), low, medium, high)
stk_prob$listing_id <- meta_test$listing_id
write_csv(stk_prob, 'results/predictions_stacking_v10.csv')

# glm stacker
ctrl <- trainControl(method = "cv", number = 3, search = "random", classProbs = TRUE,
                     summaryFunction = mnLogLoss, allowParallel = FALSE)
glm_stacker <- train(interest_level ~ ., data = meta_train, method = "glmnet",
                     metric="logLoss", trControl = ctrl, tuneLength=10)
write_rds(glm_stacker, 'models/stacker_glm.rds')
preds <- select(predict(glm_stacker, meta_test, type='prob'), low, medium, high)
preds$listing_id <- meta_test$listing_id
write_csv(preds, 'results/glm_stacker.csv')

## Ensemble predictions ----
stacknet <- read_csv('results/sigma_stack_pred.csv', col_names = c("high", "medium", "low"))
stacknet2 <- read_csv('results/sigma_stack_pred_v2.csv', col_names = c("high", "medium", "low"))
stacknet_id <- read_csv('results/test_stacknet.csv', col_names = FALSE)
stacknet$listing_id <- stacknet_id$X1
stacknet2$listing_id <- as.integer(stacknet_id$X1)
write_csv(stacknet2, 'results/stacknet2.csv')

xgb <- read_csv('results/xgb.csv')
stk <- read_csv('results/predictions_stacking_v8.csv')
lgb <- read_csv('results/lgb.csv')
glm <- read_csv('results/glm_stacker.csv')
vars <- c("low", "medium", "high", "listing_id")
tmp <- rbind(select(stk, one_of(vars)),
             #select(lgb, one_of(vars)),
             select(glm, one_of(vars)),
             #select(xgb, one_of(vars)),
             select(stacknet2, one_of(vars))
             ) %>% 
  group_by(listing_id) %>% summarise_all(mean) %>%
  mutate(listing_id = as.integer(listing_id))
write_csv(tmp, 'results/submission.csv')
