library(tidymodels)

load("data/model/model_data.rda")

decisiontree_model = decision_tree(
  cost_complexity = tune(),
  tree_depth = tune(),
  mode = "classification") %>% 
  set_engine("rpart")
decisiontree_workflow = workflow() %>% 
  add_model(decisiontree_model) %>% 
  add_recipe(model_recipe)
decisiontree_grid = grid_regular(cost_complexity(),
                                 tree_depth(),
                                 levels = 4)
decisiontree_fit = decisiontree_workflow %>% 
  tune_grid(resamples = cv_folds,
            grid = decisontree_grid,
            control = control_grid(save_pred = TRUE, verbose = TRUE),
            metrics = metric_set(roc_auc))
decisiontree_tuned = decisiontree_workflow %>% 
  finalize_workflow(select_best(decisiontree_fit, metric = "roc_auc"))
decisiontree_result = fit(decisiontree_tuned, hotel_train)
decisiontree_training_pred = 
  predict(decisiontree_result, hotel_train, type = "prob") %>% 
  bind_cols(hotel_train %>% select(is_canceled))

save(decisiontree_fit, decisiontree_result, decisiontree_training_pred,
     file = "data/model/decisiontree_model.rda")
