##Libraries
library(tidyverse)
library(tidymodels)

trainCsv <- read_csv("train.csv") %>%
  mutate(type = as.factor(type)) 

misTrainCsv <- read_csv("trainWithMissingValues.csv") %>%
  mutate(type = as.factor(type))


my_recipe <- recipe(type ~., data=trainCsv) %>% 
  step_mutate(color = as.factor(color)) %>%
  #step_dummy(all_nominal_predictors()) %>% #make dummy variables
  #step_normalize(all_numeric_predictors()) %>%
  step_impute_mean(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors())

prep <- prep(my_recipe)
baked <- bake(prep, new_data = misTrainCsv)
baked


rmse_vec(trainCsv[is.na(misTrainCsv)], baked[is.na(misTrainCsv)])



