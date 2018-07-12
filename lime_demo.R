# https://www.r-bloggers.com/explaining-complex-machine-learning-models-with-lime/


# LIBRARIES AND SOURCES ---------------------------------------------------

pacman::p_load(lime)
library(caret)
library(doParallel)
pacman::p_load(OneR)
library(janitor)
library(tidyverse)


# CONSTANTS ---------------------------------------------------------------


# DATA --------------------------------------------------------------------

data_16 <- read.table("data/2016.csv", 
                      sep = ",", header = TRUE)
data_15 <- read.table("data/2015.csv", 
                      sep = ",", header = TRUE)

common_feats <- colnames(data_16)[which(colnames(data_16) %in% colnames(data_15))]

# features and response variable for modeling
feats <- setdiff(common_feats, 
                 c("Country", "Happiness.Rank", 
                   "Happiness.Score"))
response <- "Happiness.Score"

# combine data from 2015 and 2016
data_15_16 <- rbind(select(data_15, 
                           one_of(c(feats, response))),
                    select(data_16, 
                           one_of(c(feats, response))))

data_15_16$Happiness.Score.l <- bin(data_15_16$Happiness.Score, nbins = 3, 
                                    method = "content")

data_15_16 <- select(data_15_16, -Happiness.Score) %>%
  mutate(Happiness.Score.l = plyr::revalue(Happiness.Score.l, 
                                           c("(2.83,4.79]" = "low", 
                                             "(4.79,5.89]" = "medium", 
                                             "(5.89,7.59]" = "high"))) %>% 
  as.tibble() %>% 
  clean_names()

# 1 - FIT A MODEL ---------------------------------------------------------

# configure multicore
cl <- makeCluster(detectCores())
registerDoParallel(cl)

set.seed(42)
index <- createDataPartition(data_15_16$happiness_score_l,
                             p = 0.7, list = FALSE)
train_data <- data_15_16[index, ]
test_data  <- data_15_16[-index, ]

set.seed(42)
model_mlp <- caret::train(happiness_score_l ~ .,
                          data = train_data,
                          method = "mlp",
                          trControl = trainControl(method = "repeatedcv", 
                                                   number = 10, 
                                                   repeats = 5, 
                                                   verboseIter = FALSE))


# 2 - LIME ----------------------------------------------------------------


# 2.1 - lime() function ---------------------------------------------------

# The central function of lime is lime() It creates the function that is used 
# in the next step to explain the model’s predictions.
# 
# We can give a couple of options. Check the help ?lime for details, but the
# most important to think about are:
#   
#    - Should continuous features be binned? And if so, into how many bins?
#    - How many features do we want to use in the explanatory function?
#    - How do we want to choose these features?

explanation <- lime(train_data, model_mlp, 
                    bin_continuous = TRUE, 
                    n_bins = 5, 
                    n_permutations = 1000)


# 2.2 - Cases to explain --------------------------------------------------

# Now, let’s look at how the model is explained. Here, We are not going to look 
# at all test cases but we'll' randomly choose three cases with correct 
# predictions and three with wrong predictions.

pred <- data.frame(sample_id = 1:nrow(test_data),
                   predict(model_mlp, test_data,
                           type = "prob"),
                   actual = test_data$happiness_score_l)

pred$prediction <- colnames(pred)[3:5][apply(pred[, 3:5], 
                                             1, which.max)]

pred$correct <- ifelse(pred$actual == pred$prediction, 
                       "correct", "wrong")

# We need to give our test-set data table row names with the sample names or
# IDs to be displayed in the header of our explanatory plots below
pred_cor <- filter(pred, correct == "correct")
pred_wrong <- filter(pred, correct == "wrong")

test_data_cor <- test_data %>%
  mutate(sample_id = 1:nrow(test_data)) %>%
  filter(sample_id %in% pred_cor$sample_id) %>%
  sample_n(size = 3) %>%
  remove_rownames() %>%
  tibble::column_to_rownames(var = "sample_id") %>%
  select(-happiness_score_l)

test_data_wrong <- test_data %>%
  mutate(sample_id = 1:nrow(test_data)) %>%
  filter(sample_id %in% pred_wrong$sample_id) %>%
  sample_n(size = 3) %>%
  remove_rownames() %>%
  tibble::column_to_rownames(var = "sample_id") %>%
  select(-happiness_score_l)

# 2.3 - explain() function ------------------------------------------------

# n_features: We can choose how many features we want to look at with the 
# n_features option
# 
# feature_select: To specify how we want this subset of features to be found: 
#   the default, auto, uses forward selection if we chose n_features <= 6 and 
# uses the features with highest weights otherwise. We can also directly choose 
# feature_select = "forward_selection", feature_select = "highest_weights" or 
# feature_select = "lasso_path". 

# We also want to have explanation for all three class labels in the response 
# variable (low, medium and high happiness), so I am choosing n_labels = 3.

explanation_cor <- lime::explain(test_data_cor, explanation,
                           n_labels = 3, n_features = 5)
explanation_wrong <- lime::explain(test_data_wrong, explanation,
                             n_labels = 3, n_features = 5)

# It will return a tidy tibble object that we can plot with plot_features()
plot_features(explanation_cor, ncol = 3)
plot_features(explanation_wrong, ncol = 3)

# The information in the output tibble is described in the help function ?lime 
# and can be viewed with
tibble::glimpse(explanation_cor)


# 2.4 - Case analysis example ---------------------------------------------

# So, what does this tell us, now? Let’s look at case 22 (the first row of our 
# plot for correctly predicted classes): 
# (Originally it was case 22, but something is wrong and we'll use case 10)
#
# This sample has been correctly predicted to come from the high happiness 
# group because it has:
# - a economy_gdp_per_capita > 1.275
# - a trust_government_corruption > 0.1902
# - a dystopia_residual between 2.32 and 2.68
# - a family between 1.004 and 1.119
# - a region = Middle East and Northern Africa
#
# case_2_study <- 22
case_2_study <- 10

pred %>%
  filter(sample_id == case_2_study)

train_data %>%
  gather(x, y, economy_gdp_per_capita:dystopia_residual) %>%
  ggplot(aes(x = happiness_score_l, y = y)) +
  geom_boxplot(alpha = 0.8, color = "grey") + 
  geom_point(data = gather(test_data[case_2_study, ], x, y, 
                           economy_gdp_per_capita:dystopia_residual), 
             color = "red", size = 3) +
  facet_wrap(~ x, scales = "free", ncol = 4)

# An overview over the top 5 explanatory features for case 10 is stored in:

as.data.frame(explanation_cor[1:9]) %>%
  filter(case == as.character(case_2_study))

# In a similar way, we can explore why some predictions were wrong.

analyse_case_boxplot <- function(case_2_study, pred, train_data, test_data) {
  
  print(pred %>% filter(sample_id == case_2_study))
  
  train_data %>%
    gather(x, y, economy_gdp_per_capita:dystopia_residual) %>%
    ggplot(aes(x = happiness_score_l, y = y)) +
    geom_boxplot(alpha = 0.8, color = "grey") + 
    geom_point(data = gather(test_data[case_2_study, ], x, y, 
                             economy_gdp_per_capita:dystopia_residual), 
               color = "red", size = 3) +
    facet_wrap(~ x, scales = "free", ncol = 4)
}

analyse_case_boxplot(87, pred, train_data, test_data)

