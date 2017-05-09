library(data.table)
library(lightgbm)
library(caret)
source('src/common.R')

lgb_pred <- function(model, d_test) {
  preds <- data.frame(predict(model, as.matrix(d_test), reshape=TRUE))
  preds$listing_id <- d_test$listing_id
  colnames(preds) <- c('low', 'medium', 'high', 'listing_id')
  preds[order(preds$listing_id),]
}

raw <- read_json("data/train.json")
raw_test <- read_json("data/test.json")

d <- fread('data/final_train.csv')
d_test <- fread('data/final_test.csv')
cat_vars <- c('manager_id', 'building_id', 'display_address', 'street_address', 'latlong',
              'build_bed', 'display_bed', 'street_bed', 'latlong_bed')

set.seed(1110)
train_ind <- createDataPartition(y = d$interest_level, p = .75, list = FALSE)

## Model using all features ----
d_train <- d[train_ind,]
d_val <- d[-train_ind,]

# for selecting best hyperparameters
X_train <- lgb.Dataset(as.matrix(select(d_train, -interest_level)),
                       categorical_feature = cat_vars,
                       label=to_numeric(d_train$interest_level))
X_val <- lgb.Dataset(as.matrix(select(d_val, -interest_level)),
                     categorical_feature = cat_vars,
                     label=to_numeric(d_val$interest_level))
params <- list(objective = "multiclass", metric = c("multi_logloss"), num_class = 3, num_leaves=127,
               max_depth=6)
# 0.5275 (no leakage feature), 0.5123 (with leakage), 0.51739 (LB)
model <- lgb.train(params, data=X_train,
                nrounds=2000,
                min_data=1,
                learning_rate=.03,
                feature_fraction=.5,
                early_stopping_rounds=50,
                eval_freq=10,
                valids = list(val = X_val))
# full training
X <- lgb.Dataset(as.matrix(select(d, -interest_level)),
                 categorical_feature = cat_vars,
                 label=to_numeric(d$interest_level))
model <- lgb.train(params, data=X,
                   nrounds=model$best_iter,
                   min_data=1,
                   learning_rate=0.03,
                   feature_fraction=0.5,
                   eval_freq=50,
                   valids = list(tr = X))

my_preds  <- lgb_pred(model, d_test)
write_csv(my_preds, 'results/lgb.csv')

fit_one <- function(m, feature) {
  d <- data.frame(m[1:nrow(raw),], interest_level = raw$interest_level)
  d_test <- m[-(1:nrow(raw)),]
  set.seed(1110)
  train_ind <- createDataPartition(y = d$interest_level, p = .75, list = FALSE)
  d_train <- d[train_ind,]
  d_val <- d[-train_ind,]
  
  X_train <- lgb.Dataset(as.matrix(select(d_train, one_of(c(feature)))),
                         label=to_numeric(d_train$interest_level))
  X_val <- lgb.Dataset(as.matrix(select(d_val, one_of(c(feature)))),
                       label=to_numeric(d_val$interest_level))
  params <- list(objective = "multiclass", metric = c("multi_logloss"), num_class = 3, num_leaves=127,
                 max_depth=5)
  model <- lgb.train(params, data=X_train,
                     nrounds=100,
                     min_data=1,
                     learning_rate=.1,
                     feature_fraction=1,
                     early_stopping_rounds=50,
                     eval_freq=10,
                     valids = list(val = X_val))
}
