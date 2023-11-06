##Libraries
library(tidyverse)
library(tidymodels)
library(keras)


trainCsv <- read_csv("train.csv") %>%
  mutate(type = as.factor(type)) #%>%
  #select(-id)


testCsv <- read_csv("test.csv") #%>%
  #select(-id)


#Create the recipe and bake it





nn_recipe <- recipe(type ~., data=trainCsv) %>%
  update_role(id, new_role="id") %>%
  step_mutate(color = as.factor(color)) %>%
  step_dummy(all_nominal_predictors()) %>%  #make dummy variables
  step_normalize(all_numeric_predictors()) %>% ## Turn color to factor then dummy encode color
  step_range(all_numeric_predictors(), min=0, max=1) #scale to [0,1]


prep <- prep(nn_recipe)
baked <- bake(prep, new_data = NULL)
baked



nn_model <- mlp(hidden_units = tune(),
                epochs = 50, #or 100 or 250
                activation="relu") %>%
  set_engine("keras", verbose=0) %>% #verbose = 0 prints off less
  set_mode("classification")

nn_wf <- workflow() %>%
  add_recipe(nn_recipe) %>%
  add_model(nn_model)



nn_tuneGrid <- grid_regular(hidden_units(range=c(1, 5)),
                            levels=5)

## Set up K-fold CV
folds <- vfold_cv(trainCsv, v = 3, repeats=1)

## Run the CV
CV_results <- nn_wf %>%
  tune_grid(resamples=folds,
            grid=nn_tuneGrid,
            metrics=metric_set(accuracy)) #Or leave metrics NULL

tuned_nn %>% collect_metrics() %>%
 filter(.metric=="accuracy") %>%
 ggplot(aes(x=hidden_units, y=mean)) + geom_line()

## CV tune, finalize and predict here and save results
## This takes a few min (10 on my laptop) so run it on becker if you want


collect_metrics(CV_results)


## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best("accuracy")
bestTune

## Finalize the Workflow & fit it
final_wf <- nn_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=trainCsv)

nn_predictions <- final_wf %>%
  predict(new_data = testCsv,
          type = "class")


Sub6 <- nn_predictions %>% 
  bind_cols(read_csv("test.csv")) %>% 
  select(id,.pred_class) %>%
  rename(Id= id, type = .pred_class)


write_csv(Sub5, "nnSubmission.csv")


