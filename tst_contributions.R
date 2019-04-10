

# LIBRARIES and SOURCES ---------------------------------------------------


library(lime)
library(caret)
library(janitor)
library(tidyverse)


# DATA --------------------------------------------------------------------

# ELimino species porque LIME parece tener problemas con los factores, habría
# que convertirlos a dummies
train_data <- iris %>% janitor::clean_names() %>% select(-species) %>% 
  as_tibble()


# LM MODEL ----------------------------------------------------------------

lm_fit <- lm(sepal_length ~ ., train_data)

summary(lm_fit)

# Usaremos caret porque LIME lo acepta, el lm tal cual no
lm_caret_fit <- caret::train(sepal_length ~ ., train_data,
                             method = "lm",
                             trControl = trainControl(method = "none"))

summary(lm_caret_fit)

# LIME FOR LM REGRESSION -------------------------------------------------

# PASO 1 LIME
lm_explanation <- lime(train_data, lm_caret_fit, 
                    bin_continuous = FALSE, 
                    # n_bins = 5, 
                    n_permutations = 1000)

train_pred <- tibble(sample_id = 1:nrow(train_data),
                   lm_pred = predict(lm_caret_fit),
                   actual = train_data$sepal_length)

# Esto es por si queremos trabajar solo con una muestra de train_data
train_data_probe <- train_data  %>%
  mutate(sample_id = 1:nrow(train_data)) %>%
  # sample_n(size = 3) %>%
  remove_rownames() %>%
  tibble::column_to_rownames(var = "sample_id") %>%
  select(-sepal_length)

# PASO 2 LIME
explanation_lm <- lime::explain(train_data_probe, lm_explanation,
                                n_features = 4)

# A ver qué ha salido
explanation_lm

# Son, en formato tidy, mini-modelos LM para cada observación

# Nos quedamos con lo que interesa
probe <- explanation_lm %>% 
  select(-model_type, -feature_desc) %>% 
  rename(beta_0 = model_intercept,
         betas = feature_weight,
         lime_pred = model_prediction,
         orig_pred = prediction) %>% 
  as_tibble

probe_0 <- probe %>% 
  select(case, model_r2, orig_pred, lime_pred) %>% 
  distinct()

# "Matriz de diseño"
probe_features_value <- tibble(case = probe$case, 
                              model_intercept = 1) %>% 
  inner_join(probe %>% 
               select(-model_r2, -beta_0, -data, -orig_pred, 
                      -lime_pred, -betas) %>% 
               spread(feature, feature_value) , 
             by = "case") %>% 
  distinct

# Coefs - Nótese que, como el modelo a explicar es lineal, los modelos 
# explicativos son idénticos
probe_features_coefs <- probe %>% 
  select(-model_r2, -lime_pred, -feature_value, -data, -orig_pred) %>% 
  spread(feature, betas) 

# Contributions - Nótese que y_hat == probe_0_rf$lime_pred
probe_contrib <- probe_features_value[,-1] * probe_features_coefs[,-1] 
probe_contrib <- probe_contrib %>% 
  mutate(y_hat = rowSums(probe_contrib))
probe_contrib <- probe_features_value[,1] %>% 
  bind_cols(probe_contrib)
probe_contrib

overall_contrib <- probe_contrib %>% select(-case) %>% summarise_all(mean)
overall_contrib / overall_contrib$y_hat

# Check with lm
probe_features_coefs %>% select(-case) %>% 
  summarise_all(mean)

probe_features_coefs %>% select(-case) %>% sapply(t.test) %>% t %>% 
  as.data.frame %>% 
  tibble::rownames_to_column(var = "feature") %>% 
  select(feature, p.value, conf.int, estimate )





# RF ----------------------------------------------------------------------

rf_caret_fit <- caret::train(sepal_length ~ ., train_data,
                             method = "rf",
                             trControl = trainControl(method = "repeatedcv", 
                                                      number = 10, 
                                                      repeats = 5,
                                                      verboseIter = FALSE), 
                             importance = TRUE)

rf_caret_fit

# PASO 1 LIME
rf_explanation <- lime(train_data, rf_caret_fit, 
                       bin_continuous = FALSE, 
                       # n_bins = 5, 
                       n_permutations = 1000)

# PASO 2 LIME
explanation_rf <- lime::explain(train_data_probe, rf_explanation,
                                n_features = 4)
# A ver qué ha salido
explanation_rf

# Nos quedamos con lo que interesa
probe_rf <- explanation_rf %>% 
  select(-model_type, -feature_desc) %>% 
  rename(beta_0 = model_intercept,
         betas = feature_weight,
         lime_pred = model_prediction,
         orig_pred = prediction) %>% 
  as_tibble

probe_0_rf <- probe_rf %>% 
  select(case, model_r2, orig_pred, lime_pred) %>% 
  distinct()

# "Matriz de diseño"
probe_features_value_rf <- tibble(case = probe_rf$case, 
                               model_intercept = 1) %>% 
  inner_join(probe_rf %>% 
               select(-model_r2, -beta_0, -data, -orig_pred, 
                      -lime_pred, -betas) %>% 
               spread(feature, feature_value) , 
             by = "case") %>% 
  distinct

# Coefs - Nótese que, como el modelo a explicar no es lineal, los modelos 
# explicativos no son iguales
probe_features_coefs_rf <- probe_rf %>% 
  select(-model_r2, -lime_pred, -feature_value, -data, -orig_pred) %>% 
  spread(feature, betas) 

# Contributions - Nótese que y_hat == probe_0_rf$lime_pred
probe_contrib_rf <- probe_features_value_rf[,-1] * probe_features_coefs_rf[,-1] 
probe_contrib_rf <- probe_contrib_rf %>% 
  mutate(y_hat = rowSums(probe_contrib_rf))
probe_contrib_rf <- probe_features_value_rf[,1] %>% 
  bind_cols(probe_contrib_rf)
probe_contrib_rf

overall_contrib_rf <- probe_contrib_rf %>% select(-case) %>% summarise_all(mean)
overall_contrib_rf / overall_contrib_rf$y_hat

# COMENTARIOS
# 1 - Los signos coinciden con los del modelo lineal
# 2 - Aquí no tiene sentido la contribución de model_intercept
#
# La solución podría ser la de prophet: repartir proporcionalmente la 
# contribución de model_intercept entre los otros predictores

features <- predictors(rf_caret_fit)

new_probe_contrib_rf <- probe_contrib_rf %>% 
  mutate(sum_predictors = rowSums(select(., features))) %>% 
  mutate_at(vars(features),
            funs(. / sum_predictors * y_hat )) %>% 
  select(-model_intercept, -sum_predictors)

# Y, ahora si, los aportes serían:
overall_contrib_rf <- new_probe_contrib_rf %>% select(-case) %>% summarise_all(mean)
overall_contrib_rf / overall_contrib_rf$y_hat

# Comparemos con la importancia de variables
varImp(rf_caret_fit, scale = F)

# Aportes en valor absoluto
perc_contrib_rf <- (overall_contrib_rf / overall_contrib_rf$y_hat) %>% 
  mutate_at(vars(features), funs(abs(.))) %>% 
  # mutate(abs_y_hat = rowSums(select(., features) %>% 
  #                              mutate_at(vars(features), funs(abs(.)))))
  mutate(abs_y_hat = rowSums(select(., features)))

perc_contrib_rf / perc_contrib_rf$abs_y_hat
