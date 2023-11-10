library(bonsai)
library(lightgbm)
library(tidymodels)
library(tidyverse)

##Read in the datasets

trainCsv <- read_csv("train.csv") %>%
  mutate(type = as.factor(type)) #%>%
#select(-id)


testCsv <- read_csv("test.csv") #%>%
#select(-id)


#Create the recipe and bake it


bart_recipe <- recipe(type ~., data=trainCsv) %>%
  update_role(id, new_role="id") %>%
  step_mutate(color = as.factor(color)) %>%
  step_dummy(all_nominal_predictors()) %>%  #make   dummy variables
  step_normalize(all_numeric_predictors())


prep <- prep(bart_recipe)
baked <- bake(prep, new_data = NULL)
baked



bart_model <- bart(trees=tune()) %>% # BART figures out depth and learn_rate
  set_engine("dbarts") %>% # might need to install
  set_mode("classification")


bart_wf <- workflow() %>%
  add_recipe(bart_recipe) %>%
  add_model(bart_model)



bart_tuneGrid <- grid_regular(trees(),
                               levels=10)

## Set up K-fold CV
folds <- vfold_cv(trainCsv, v = 3, repeats=1)

## Run the CV
CV_results <- bart_wf %>%
  tune_grid(resamples=folds,
            grid=bart_tuneGrid,
            metrics=metric_set(accuracy)) #Or leave metrics NULL

CV_results %>% collect_metrics() %>%
  filter(.metric=="accuracy") %>%
  ggplot(aes(x=trees, y=mean)) + geom_line()

## CV tune, finalize and predict here and save results
## This takes a few min (10 on my laptop) so run it on becker if you want


collect_metrics(CV_results)


## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best("accuracy")
bestTune

## Finalize the Workflow & fit it
final_wf <- bart_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=trainCsv)

bart_predictions <- final_wf %>%
  predict(new_data = testCsv,
          type = "class")


Sub7 <- bart_predictions %>% 
  bind_cols(read_csv("test.csv")) %>% 
  select(id,.pred_class) %>%
  rename(Id= id, type = .pred_class)


#Writes csv
write_csv(Sub7, "bartSubmission.csv")

