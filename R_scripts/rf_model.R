library(tidymodels)
library(treeshap)

load("data/model/model_data.rda")

rf_model = rand_forest(
  min_n = tune(),
  mtry = tune(),
  trees = tune(),
  mode = "classification") %>%
  set_engine("ranger")
rf_workflow = workflow() %>% 
  add_model(rf_model) %>% 
  add_recipe(model_recipe)
rf_grid = grid_latin_hypercube(
  min_n(), 
  mtry(range = c(4, 12)), 
  trees(), 
  size = 50)
rf_fit = rf_workflow %>% 
  tune_grid(resamples = cv_folds,
            grid = rf_grid,
            control = control_grid(save_pred = TRUE, verbose = TRUE),
            metrics = metric_set(roc_auc))
rf_tuned = rf_workflow %>%
  update_model(rand_forest(mtry = 8, min_n = 3, trees = 665) %>% 
                 set_engine("ranger", importance = "impurity") %>% 
                 set_mode("classification"))
rf_result = fit(rf_tuned, hotel_train)
rf_training_pred = 
  predict(rf_result, hotel_train, type = "prob") %>% 
  bind_cols(hotel_train %>% select(is_canceled))
save(rf_fit, rf_result, rf_training_pred, file = "data/model/rf_model.rda")

# get final model
X = model.matrix(~.-1, data = rf_result$pre$mold$predictors)
colnames(X) = make.names(colnames(X))
trn = as.data.frame(X)
trn$is_canceled=as.numeric(rf_result$pre$mold$outcomes$is_canceled)-1
names(trn) = make.names(names(trn))
rfo = ranger::ranger(is_canceled ~ ., 
                     data = trn,
                     mtry=8, num.trees=665, min.node.size = 3, num.threads = 8)
model_unified = ranger.unify(rfo, trn %>% select(-is_canceled))
treeshap_res = treeshap(unified_model=model_unified, x=as.data.frame(X)[1:100,])
save(rfo, treeshap_res, file = "data/model/rf_final_model.rda")

