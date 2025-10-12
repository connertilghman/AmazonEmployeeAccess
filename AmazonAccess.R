library(tidyverse)
library(tidymodels)
library(vroom)
library(patchwork)
library(dplyr)
library(DataExplorer)
library(glmnet)
sample <- vroom("sampleSubmission.csv")
test <- vroom("test.csv")
train <- vroom("train.csv")
summary(train)

train <- train |>
  mutate(ACTION = as.factor(ACTION))


ggplot(train, aes(factor(ACTION))) +
  geom_bar(fill = "steelblue") +
  labs(title = "Access Decision Distribution", x = "ACTION", y = "Count")


create_report(train)

my_recipe <- recipe(ACTION ~ ., data = train) |>
  step_mutate_at(all_numeric_predictors(), fn = factor) |>
  step_other(all_nominal_predictors(), threshold = 0.001) |>
  step_dummy(all_nominal_predictors())

prep_rec <- prep(my_recipe)
baked_data <- bake(prep_rec, new_data = train)
cat(ncol(baked_data))

logRegModel <- logistic_reg() |>
  set_engine("glm")

logReg_workflow <- workflow() |>
  add_model(logRegModel) |>
  add_recipe(my_recipe)

logReg_fit <- fit(logReg_workflow, data = train)

amazon_predictions <- predict(logReg_fit,
                              new_data=test,
                              type="prob")
kaggle_submission <- test|>
  bind_cols(amazon_predictions)|>
  select(id, .pred_1) |>
  rename(Action=.pred_1) |>
  rename(Id=id)

vroom_write(x=kaggle_submission, file="./AmazonPreds.csv", delim=",")
