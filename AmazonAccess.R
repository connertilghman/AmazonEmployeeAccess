library(tidyverse)
library(patchwork)
library(tidymodels)
library(embed)
library(vroom)
library(discrim)
library(kernlab)
library(themis)

sample <- vroom("sampleSubmission.csv")
test <- vroom("test.csv")
train <- vroom("train.csv")
summary(train)


train <- train |>
  mutate(ACTION = as.factor(ACTION))


# ggplot(train, aes(factor(ACTION))) +
#   geom_bar(fill = "steelblue") +
#   labs(title = "Access Decision Distribution", x = "ACTION", y = "Count")
# 
# 
# create_report(train)

my_recipe <- recipe(ACTION ~ ., data = train) |>
  step_mutate_at(all_numeric_predictors(), fn = factor) |>
  #step_other(all_nominal_predictors(), threshold = .001) %>% 
  # combines categorical values that occur
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) |>
  step_normalize(all_predictors()) #target encoding
  #step_pca(all_predictors(), threshold=.9) #Threshold is between 0 and 1
# step_smote(all_outcomes(), neighbors=5)

prep_rec <- prep(my_recipe)
baked_data <- bake(prep_rec, new_data = train)
cat(ncol(baked_data))

# logRegModel <- logistic_reg() |>
#   set_engine("glm")
# 
# logReg_workflow <- workflow() |>
#   add_model(logRegModel) |>
#   add_recipe(my_recipe)
# 
# logReg_fit <- fit(logReg_workflow, data = train)
# 
# amazon_predictions <- predict(logReg_fit,
#                               new_data=test,
#                               type="prob")

# my_mod <- logistic_reg(mixture=tune(), penalty=tune()) |>
#   set_engine("glmnet")

forest_mod <- rand_forest(mtry = tune(),
                      min_n=tune(),
                      trees=1000) |>
  set_engine("ranger") |>
  set_mode("classification")

n_cores <- parallel::detectCores() - 1
cl <- makeCluster(n_cores)
registerDoParallel(cl)

# knn_mod <- nearest_neighbor(
#   neighbors = tune()) |>
#   set_mode("classification") |>
#   set_engine("kknn")
# 
# nb_mod <- naive_Bayes(Laplace=tune(), smoothness=tune()) |>
#   set_mode("classification") |>
#   set_engine("naivebayes") 
# 
# nn_model <- mlp(hidden_units = tune(),
#                 epochs = 50 )|>
#   set_engine("keras") |>
#   set_mode("classification")


svmRadial <- svm_rbf(rbf_sigma=tune(), cost=tune()) |>
  set_mode("classification") |>
  set_engine("kernlab")

amazon_workflow <- workflow() |>
  add_recipe(my_recipe) |>
  add_model(forest_mod)

tuning_grid <- grid_regular(mtry(range = c(1,9)),
                            min_n(),
                            levels=5)

folds <- vfold_cv(train, v=10, repeats=1)

control <- control_grid(save_pred = FALSE, save_workflow = FALSE, verbose = FALSE)

CV_results <- amazon_workflow |> 
  tune_grid(
    resamples=folds,
    grid=tuning_grid,
    metrics=metric_set(roc_auc),
    control = control)


# graph <- CV_results |> 
#   collect_metrics() |>
#   filter(.metric=="accuracy") |>
#   ggplot(aes(x=hidden_units, y=mean)) + geom_line()

bestTune <- CV_results |>
  select_best(metric = "roc_auc")

final_wf <- amazon_workflow |>
  finalize_workflow(bestTune) |>
  fit(data=train)

forest_preds <- predict(final_wf, new_data = test, type="prob")

kaggle_submission <- test|>
  bind_cols(forest_preds)|>
  select(id, .pred_1) |>
  rename(Action=.pred_1) |>
  rename(Id=id)

vroom_write(x=kaggle_submission, file="./NewForestPreds.csv", delim=",")

stopCluster(cl)
registerDoSEQ()

