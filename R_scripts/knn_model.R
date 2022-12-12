library(tidymodels)

load("data/model/model_data.rda")

knn_model = nearest_neighbor(
  neighbors = tune(),
  mode = "classification") %>% 
  set_engine("kknn")
knn_workflow = workflow() %>% 
  add_model(knn_model) %>% 
  add_recipe(model_recipe)
knn_params = extract_parameter_set_dials(knn_model)
knn_grid = data.frame(neighbors = c(10,20,30,40,50,60))
knn_fit = knn_workflow %>% 
  tune_grid(resamples = cv_folds,
            grid = knn_grid,
            control = control_grid(save_pred = TRUE, verbose = TRUE),
            metrics = metric_set(roc_auc))
knn_tuned = knn_workflow %>% 
  finalize_workflow(select_best(knn_fit, metric = "roc_auc"))
knn_result = fit(knn_tuned, hotel_train)
knn_training_pred = 
  predict(knn_result, hotel_train, type = "prob") %>% 
  bind_cols(hotel_train %>% select(is_canceled))

save(knn_fit, knn_result, knn_training_pred, file = "data/model/knn_model.rda")
