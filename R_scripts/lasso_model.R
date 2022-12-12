library(tidymodels)

load("data/model/model_data.rda")

lasso_model = 
  logistic_reg(penalty = tune(), mixture = 1) %>% 
  set_engine("glmnet")
lasso_workflow = workflow() %>% add_model(lasso_model) %>% add_recipe(model_recipe)
lasso_grid = tibble(penalty = 10^seq(-4, -1, length.out = 20))
lasso_fit = lasso_workflow %>% 
  tune_grid(resamples = cv_folds,
            grid = lasso_grid,
            control = control_grid(save_pred = TRUE, verbose = TRUE),
            metrics = metric_set(roc_auc))
lasso_tuned = lasso_workflow %>% 
  finalize_workflow(select_best(lasso_fit, metric = "roc_auc"))
lasso_result = fit(lasso_tuned, hotel_train)
lasso_training_pred = 
  predict(lasso_result, hotel_train, type = "prob") %>% 
  bind_cols(hotel_train %>% select(is_canceled))

save(lasso_fit, lasso_result, lasso_training_pred, 
     file = "data/model/lasso_model.rda")

