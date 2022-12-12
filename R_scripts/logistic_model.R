library(tidymodels)

load("data/model/model_data.rda")

lr_model =
  logistic_reg() %>% 
  set_engine("glm")
lr_wflow = workflow() %>% 
  add_model(lr_model) %>% 
  add_recipe(model_recipe)
lr_fit = lr_wflow %>% 
  tune_grid(hotel_train, 
            resamples = cv_folds,
            control = control_grid(save_pred = TRUE, verbose = TRUE),
            metrics = metric_set(roc_auc))
lr_tuned = lr_wflow %>% 
  finalize_workflow(select_best(lr_fit, metric = "roc_auc"))
lr_result = fit(lr_tuned, hotel_train)
lr_training_pred = 
  predict(lr_result, hotel_train, type = "prob") %>% 
  bind_cols(hotel_train %>% select(is_canceled))

save(lr_fit, lr_result, lr_training_pred, file = "data/model/logistic_model.rda")

