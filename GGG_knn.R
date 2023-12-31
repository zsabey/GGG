##Libraries
library(tidyverse)
library(tidymodels)


trainCsv <- read_csv("train.csv") %>%
  mutate(type = as.factor(type)) %>%
  select(-id)


testCsv <- read_csv("test.csv") %>%
  select(-id)


#Create the recipe and bake it

knn_recipe <- recipe(type ~., data=trainCsv) %>%
  step_mutate(color = as.factor(color)) %>%
  step_dummy(all_nominal_predictors()) %>%  #make dummy variables
  step_normalize(all_numeric_predictors()) %>%
  step_pca(all_predictors(),threshold = .9)


prep <- prep(knn_recipe)
baked <- bake(prep, new_data = NULL)
baked


## knn model
knn_model <- nearest_neighbor(neighbors=tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kknn")

knn_wf <- workflow() %>%
  add_recipe(knn_recipe) %>%
  add_model(knn_model)

## Fit or Tune Model HERE


## Tune smoothness and Laplace here

## Set up grid of tuning values
tuning_grid <- grid_regular(neighbors(),
                            levels = 10) ## L^2 total tuning possibilities


## Set up K-fold CV
folds <- vfold_cv(trainCsv, v = 3, repeats=1)

## Run the CV
CV_results <- knn_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(accuracy)) #Or leave metrics NULL

## Find best tuning parameters
collect_metrics(CV_results) %>% # Gathers metrics into DF
  filter(.metric=="accuracy") %>%
  ggplot(data=., aes(x=neighbors, y=mean))+ #, color=factor(smoothness))) +
  geom_line()

collect_metrics(CV_results)


## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best("accuracy")
bestTune

## Finalize the Workflow & fit it
final_wf <- knn_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=trainCsv)

knn_predictions <- final_wf %>%
  predict(new_data = testCsv,
          type = "class")


Sub5 <- knn_predictions %>% 
  bind_cols(read_csv("test.csv")) %>% 
  select(id,.pred_class) %>%
  rename(Id= id, type = .pred_class)


write_csv(Sub5, "knnSubmission.csv")
