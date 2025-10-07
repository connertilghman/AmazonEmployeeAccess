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


ggplot(train, aes(factor(ACTION))) +
  geom_bar(fill = "steelblue") +
  labs(title = "Access Decision Distribution", x = "ACTION", y = "Count")


create_report(train)

my_recipe <- recipe(ACTION ~ ., data = train) |>
  step_mutate_at(all_nominal_predictors(), fn = factor) |>
  step_other(all_nominal_predictors(), threshold = 0.001) |>
  step_dummy(all_nominal_predictors())

prep_rec <- prep(my_recipe)
baked_data <- bake(prep_rec, new_data = train)
