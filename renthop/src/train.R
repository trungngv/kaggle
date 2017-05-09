rm(list = ls())
library(caret)
library(pROC)
library(tm)
source('src/pre_process.R')

d <- fread('data/final_train.csv')
d_test <- fread('data/final_test.csv')

## Train models ---- 
set.seed(1110)
train_ind <- createDataPartition(y = d$interest_level, p = .75, list = FALSE)
d_train <- d[train_ind,]
d_val <- d[-train_ind,]

## Train xgb model ----
# This data does not have features that were derived using labels so we're free to train on FULL data
# set
registerDoParallel(4)
getDoParWorkers()
params <- list(eta = .03, gamma = 1, max_depth = 4, min_child_weight = 1,
               subsample = .7, colsample_bytree = .5,
               num_class = 3, objective = "multi:softprob", eval_metric = "mlogloss")
xgb_d_train <- xgb.DMatrix(data.matrix(select(d_train, -interest_level)),
                            label=to_numeric(d_train$interest_level))
xgb_d_val <- xgb.DMatrix(as.matrix(select(d_val, -interest_level)),
                          label=to_numeric(d_val$interest_level))
# choosing hyperparameters
# 0.527128, eta = .03, nrounds=1510 (all features)
# 0.525906, eta = .03, nrounds=20006 (new features)
# 0.5267 (col=.7)
xgb_model <- xgb.train(params, xgb_d_train,
                       nrounds = 3000, verbose = TRUE
                       , early.stop.round = 50
                       , watchlist = list(val = xgb_d_val, tr = xgb_d_train)
)

imp <- xgb.importance(feature_names = colnames(select(d, -interest_level)),
                             model=xgb_model)
View(imp)

## evaluation on val_prob ----

val_prob <- matrix(predict(xgb_model, xgb_d_val), ncol = 3, byrow=T) %>% data.frame()
colnames(val_prob) <- c("low", "medium", "high")
val_prob <- select(val_prob, high, low, medium)
val_label <- probs_to_label(val_prob)$label
logloss(d_val$interest_level, val_prob)
confusionMatrix(d_val$interest_level, val_label)

# full training ----
xgb_d <- xgb.DMatrix(data.matrix(select(d, -interest_level)),
                     label=to_numeric(d$interest_level))
xgb_d_test <- xgb.DMatrix(as.matrix(d_test))
xgb_model <- xgb.train(params, xgb_d,
                       nrounds = xgb_model$bestInd, verbose = TRUE
                       , early.stop.round = 50
                       , watchlist =list(tr = xgb_d)
)
write_rds(xgb_model, 'models/xgb.rds')
preds <- matrix(predict(xgb_model, xgb_d_test), ncol = 3, byrow=T) %>% data.frame()
colnames(preds) <- c("low", "medium", "high")
preds$listing_id <- d_test$listing_id
write_csv(preds, 'results/xgb.csv')
