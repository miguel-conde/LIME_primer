

train_data <- iris %>% janitor::clean_names() %>% select(-species)

lm_fit <- lm(sepal_length ~ ., train_data)

summary(lm_fit)

lm_caret_fit <- caret::train(sepal_length ~ ., train_data,
                             method = "lm",
                             trControl = trainControl(method = "none"))

lm_explanation <- lime(train_data, lm_caret_fit, 
                    bin_continuous = FALSE, 
                    # n_bins = 5, 
                    n_permutations = 1000)

train_pred <- data.frame(sample_id = 1:nrow(train_data),
                   pred = predict(lm_caret_fit, train_data),
                   actual = train_data$sepal_length)

train_data_probe <- train_data  %>%
  mutate(sample_id = 1:nrow(train_data)) %>%
  # sample_n(size = 3) %>%
  remove_rownames() %>%
  tibble::column_to_rownames(var = "sample_id") %>%
  select(-sepal_length)

explanation_lm <- lime::explain(train_data_probe, lm_explanation,
                                n_features = 4)

probe <- explanation_lm %>% 
  select(-model_type, -feature_desc, -data, -prediction) %>% 
  as_tibble

probe_0 <- probe %>% 
  select(case, model_r2, model_prediction) %>% 
  distinct()

probe_feature_value <- tibble(case = probe_0$case, 
                              model_intercept = 1) %>% 
  inner_join(probe %>% 
               select(-model_r2, -model_intercept, -model_prediction, -feature_weight) %>% 
               spread(feature, feature_value) , 
             by = "case")

# Coefs
probe_feature_weight <- probe %>% 
  select(-model_r2, -model_prediction, -feature_value) %>% 
  spread(feature, feature_weight) 

# Contributions
probe_prediction <- probe_feature_value[,-1] * probe_feature_weight[,-1] 
probe_prediction <- probe_prediction %>% 
  mutate(y_hat = rowSums(probe_prediction))
probe_prediction <- probe_feature_value[,1] %>% 
  bind_cols(probe_prediction)

overall_contrib <- probe_prediction %>% select(-case) %>% summarise_all(mean)
overall_contrib / overall_contrib$y_hat

# Check with lm
probe_feature_weight %>% select(-case) %>% 
  summarise_all(mean)

probe_feature_weight %>% select(-case) %>% sapply(t.test) %>% t %>% 
  as.data.frame %>% 
  tibble::rownames_to_column(var = "feature") %>% 
  select(feature, p.value, conf.int, estimate )


# RF ----------------------------------------------------------------------

rf_caret_fit <- caret::train(sepal_length ~ ., train_data,
                             method = "rf",
                             trControl = trainControl(method = "repeatedcv", 
                                                      number = 10, 
                                                      repeats = 5, 
                                                      verboseIter = FALSE))

rf_explanation <- lime(train_data, rf_caret_fit, 
                       bin_continuous = FALSE, 
                       # n_bins = 5, 
                       n_permutations = 1000)

# train_pred <- data.frame(sample_id = 1:nrow(train_data),
#                          pred = predict(rf_caret_fit, train_data),
#                          actual = train_data$sepal_length)
# 
# train_data_probe <- train_data  %>%
#   mutate(sample_id = 1:nrow(train_data)) %>%
#   # sample_n(size = 3) %>%
#   remove_rownames() %>%
#   tibble::column_to_rownames(var = "sample_id") %>%
#   select(-sepal_length)

explanation_rf <- lime::explain(train_data_probe, rf_explanation,
                                n_features = 4)

rf_probe <- explanation_rf %>% 
  select(-model_type, -feature_desc, -data, -prediction) %>% 
  as_tibble

rf_probe_0 <- probe %>% 
  select(case, model_r2, model_prediction) %>% 
  distinct()

rf_probe_feature_value <- tibble(case = rf_probe_0$case, 
                              model_intercept = 1) %>% 
  inner_join(rf_probe %>% 
               select(-model_r2, -model_intercept, -model_prediction, -feature_weight) %>% 
               spread(feature, feature_value), 
             by = "case")

# Coefs
rf_probe_feature_weight <- rf_probe %>% 
  select(-model_r2, -model_prediction, -feature_value) %>% 
  spread(feature, feature_weight) %>% 
  mutate(case = as.numeric(case)) %>% arrange(case)

rf_probe_feature_weight <- rf_probe_feature_weight %>% 
  mutate_at(vars(petal_length, petal_width, sepal_width),
            funs(
              . + . / (petal_length + petal_width + sepal_width) * model_intercept
            ))

# Contributions
rf_probe_prediction <- rf_probe_feature_value[,-1] * rf_probe_feature_weight[,-1] 
rf_probe_prediction <- rf_probe_prediction %>% 
  mutate(y_hat = rowSums(rf_probe_prediction))
rf_probe_prediction <- rf_probe_feature_value[,1] %>% 
  bind_cols(rf_probe_prediction)

rf_overall_contrib <- rf_probe_prediction %>% select(-case) %>% summarise_all(mean)
rf_overall_contrib / rf_overall_contrib$y_hat

# Check with lm
rf_probe_feature_weight %>% select(-case) %>% 
  summarise_all(mean)

rf_probe_feature_weight %>% select(-case) %>% sapply(t.test) %>% t %>% 
  as.data.frame %>% 
  tibble::rownames_to_column(var = "feature") %>% 
  select(feature, p.value, conf.int, estimate )

plot(train_data$sepal_length, predict(rf_caret_fit),
     xlim = c(0, 8), ylim = c(0, 8))
abline(lm_fit)
abline(a = 5.310096, b = 0.09219901, col = "blue") 
abline(a = 0, b = 3.882815, col = "red")
