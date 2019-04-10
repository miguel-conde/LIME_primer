

get_lime_contributions <- function(model_fit, 
                                   train_data, 
                                   new_data,
                                   features) {
  
  n_features = length(features)
  
  # PASO 1 LIME
  explainer <- lime(train_data, model_fit, 
                    bin_continuous = FALSE, 
                    n_permutations = 1000)
  # PASO 2 LIME
  explanation <- lime::explain(new_data, explainer,
                               n_features = n_features)
  
  # Nos quedamos con lo que interesa
  probe <- explanation %>% 
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
  
  # Coefs 
  probe_features_coefs <- probe %>% 
    select(-model_r2, -lime_pred, -feature_value, -data, -orig_pred) %>% 
    spread(feature, betas) 
  
  # "Raw" Contributions 
  probe_contrib <- probe_features_value[,-1] * probe_features_coefs[,-1] 
  probe_contrib <- probe_contrib %>% 
    mutate(y_hat = rowSums(probe_contrib))
  probe_contrib <- probe_features_value[,1] %>% 
    bind_cols(probe_contrib)
  
  new_probe_contrib <- probe_contrib %>% 
    mutate(sum_predictors = rowSums(select(., features)),
           orig_pred = probe_0$orig_pred) %>% 
    # Reescalamos para que las aportaciones sumen la predicción del modelo
    mutate_at(vars(features),
              funs(. / sum_predictors * orig_pred )) %>% 
    select(-model_intercept, -sum_predictors, -y_hat)
  
  return(new_probe_contrib)
  
}