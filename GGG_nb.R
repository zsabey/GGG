##Libraries
library(tidyverse)
library(tidymodels)
library(naivebayes)
library(discrim)
library(embed)

trainCsv <- read_csv("train.csv") 


testCsv <- read_csv("test.csv") 


#Create the recipe and bake it

nb_recipe <- recipe(type ~., data=trainCsv) %>%
  step_rm(id) %>%
  step_lencode_glm(color, outcome = vars(type))


prep <- prep(nb_recipe)
baked <- bake(prep, new_data = NULL)
baked


## nb model
nb_model <- naive_Bayes(Laplace=tune(), smoothness = tune()) %>% # set or tune
  set_engine("naivebayes") %>%
  set_mode("classification")

nb_wf <- workflow() %>%
  add_recipe(nb_recipe) %>%
  add_model(nb_model)

## Fit or Tune Model HERE


## Tune smoothness and Laplace here

## Set up grid of tuning values
tuning_grid <- grid_regular(Laplace(),
                            smoothness(),
                            levels = 5) ## L^2 total tuning possibilities


## Set up K-fold CV
folds <- vfold_cv(trainCsv, v = 5, repeats=1)

## Run the CV
CV_results <- nb_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(accuracy)) #Or leave metrics NULL

## Find best tuning parameters
collect_metrics(CV_results) %>% # Gathers metrics into DF
  filter(.metric=="accuracy") %>%
  ggplot(data=., aes(x=Laplace, y=mean))+ #, color=factor(smoothness))) +
  geom_line()

collect_metrics(CV_results)


## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best("accuracy")
bestTune


## Finalize the Workflow & fit it
final_wf <- nb_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=trainCsv)

nb_predictions <- final_wf %>%
  predict(new_data = testCsv,
          type = "class")


Sub5 <- nb_predictions %>% 
  bind_cols(read_csv("test.csv")) %>% 
  select(id,.pred_class) %>%
  rename(Id= id, type = .pred_class)


write_csv(Sub5, "nbSubmission.csv")
