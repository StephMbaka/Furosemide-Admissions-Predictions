# ----- Packages -----
library(readxl)
library(dplyr)
library(tidyr)
library(caret)
library(xgboost)
library(pROC)             
library(SHAPforxgboost)
library(ggplot2)
library(readr)

# ------------------- Data prep -------------------

# 1) Load full dataset
raw_df <- completedata_NoNAs

# 2) Keep clinically relevant predictors + outcome + patient ID
keep_cols <- c(
  "patID", "admitted",
  "age_at_event", "sex", "bmi", "smoking", "alcohol", "ethnicgroup", "IMD",
  "NumberofComobidites", "Year_sinceHF",
  "Total_Yearly_Dose"
)

# 3) Subset & basic cleaning
df <- raw_df %>%
  select(all_of(keep_cols)) %>%
  mutate(
    admitted = factor(admitted, levels = c("No", "Yes")),
    sex       = factor(sex),
    smoking   = factor(smoking),
    alcohol   = factor(alcohol),
    ethnicgroup = factor(ifelse(is.na(ethnicgroup), "Other/Unknown", as.character(ethnicgroup))),
    age_at_event       = suppressWarnings(as.numeric(age_at_event)),
    bmi                = suppressWarnings(as.numeric(bmi)),
    NumberofComobidites= suppressWarnings(as.numeric(NumberofComobidites)),
    Year_sinceHF       = suppressWarnings(as.numeric(Year_sinceHF)),
    Total_Yearly_Dose  = suppressWarnings(as.numeric(Total_Yearly_Dose))
  ) %>%
  drop_na(admitted, patID, Total_Yearly_Dose, age_at_event, bmi,
          NumberofComobidites, sex, smoking, alcohol)

# ------------------- Feature setup -------------------

if ("Year_sinceHF" %in% names(df)) {
  df <- df %>% dplyr::filter(Year_sinceHF %in% 0:10)
}

feat_df <- df %>%
  dplyr::select(-patID, -admitted) %>%
  dplyr::mutate(dplyr::across(where(is.character), as.factor)) %>%
  dplyr::select(where(function(col) !(is.factor(col) && nlevels(col) < 2))) %>%
  tidyr::drop_na()

keep_rows <- as.integer(rownames(feat_df))
df_kept   <- df[keep_rows, ]
pid       <- df_kept$patID

X <- model.matrix(~ . - 1, data = feat_df)
y <- ifelse(df_kept$admitted == "Yes", 1, 0)

# ------------------- CV setup -------------------

set.seed(123)
K <- 5
uniq_pid <- unique(pid)
fold_id_per_patient <- sample(rep(1:K, length.out = length(uniq_pid)))
names(fold_id_per_patient) <- as.character(uniq_pid)
row_fold <- fold_id_per_patient[as.character(pid)]
train_index_list <- lapply(1:K, function(k) which(row_fold != k))
test_index_list  <- lapply(1:K, function(k) which(row_fold == k))

# ------------------- Hyperparameter grid -------------------

grid <- expand.grid(
  eta               = c(0.05, 0.1),
  max_depth         = c(3, 5),
  gamma             = c(0, 1),
  subsample         = c(0.7, 0.9),
  colsample_bytree  = c(0.7, 1.0),
  min_child_weight  = c(1, 3)
)

nrounds_max <- 3000
early_stopping_rounds <- 50

cv_results <- list()

for (g in seq_len(nrow(grid))) {
  params <- list(
    booster          = "gbtree",
    objective        = "binary:logistic",
    eval_metric      = "auc",
    eta              = grid$eta[g],
    max_depth        = grid$max_depth[g],
    gamma            = grid$gamma[g],
    subsample        = grid$subsample[g],
    colsample_bytree = grid$colsample_bytree[g],
    min_child_weight = grid$min_child_weight[g]
  )
  
  fold_metrics <- data.frame()
  
  for (k in seq_along(train_index_list)) {
    tr_idx <- train_index_list[[k]]
    te_idx <- test_index_list[[k]]
    
    X_tr <- X[tr_idx, , drop = FALSE]; y_tr <- y[tr_idx]
    X_te <- X[te_idx, , drop = FALSE]; y_te <- y[te_idx]
    
    pos <- sum(y_tr == 1); neg <- sum(y_tr == 0)
    spw <- ifelse(pos > 0, neg / pos, 1)
    
    dtrain <- xgb.DMatrix(data = X_tr, label = y_tr)
    dvalid <- xgb.DMatrix(data = X_te, label = y_te)
    
    fit <- xgb.train(
      params = c(params, list(scale_pos_weight = spw)),
      data   = dtrain,
      nrounds = nrounds_max,
      watchlist = list(valid = dvalid),
      early_stopping_rounds = early_stopping_rounds,
      verbose = 0
    )
    
    pred <- predict(fit, newdata = dvalid, ntreelimit = fit$best_iteration)
    roc_obj <- pROC::roc(response = y_te, predictor = pred, quiet = TRUE)
    auc_val <- as.numeric(pROC::auc(roc_obj))
    
    pred_cls <- ifelse(pred >= 0.5, 1, 0)
    TP <- sum(pred_cls == 1 & y_te == 1)
    TN <- sum(pred_cls == 0 & y_te == 0)
    FP <- sum(pred_cls == 1 & y_te == 0)
    FN <- sum(pred_cls == 0 & y_te == 1)
    sens <- ifelse((TP + FN) > 0, TP/(TP + FN), NA)
    spec <- ifelse((TN + FP) > 0, TN/(TN + FP), NA)
    
    fold_metrics <- rbind(fold_metrics,
                          data.frame(fold = k, auc = auc_val,
                                     sens_50 = sens, spec_50 = spec,
                                     best_iter = fit$best_iteration))
  }
  
  cv_results[[g]] <- list(
    params = params,
    metrics = fold_metrics,
    mean_auc = mean(fold_metrics$auc, na.rm = TRUE),
    mean_sens = mean(fold_metrics$sens_50, na.rm = TRUE),
    mean_spec = mean(fold_metrics$spec_50, na.rm = TRUE),
    mean_best_iter = round(mean(fold_metrics$best_iter))
  )
  
  cat(sprintf("Grid %d/%d  |  AUC=%.3f  Sens=%.3f  Spec=%.3f  best_iter~%d\n",
              g, nrow(grid), cv_results[[g]]$mean_auc,
              cv_results[[g]]$mean_sens, cv_results[[g]]$mean_spec,
              cv_results[[g]]$mean_best_iter))
}

# ------------------- Final model -------------------

scores <- sapply(cv_results, function(z) z$mean_auc)
best_id <- which.max(scores)
best <- cv_results[[best_id]]
cat("\nSelected params (by AUC):\n"); print(best$params)

pos_all <- sum(y == 1); neg_all <- sum(y == 0)
spw_all <- ifelse(pos_all > 0, neg_all / pos_all, 1)

dall <- xgb.DMatrix(data = X, label = y)
final_model <- xgb.train(
  params = c(best$params, list(scale_pos_weight = spw_all)),
  data   = dall,
  nrounds = max(50, best$mean_best_iter),
  verbose = 0
)

# ------------------- SHAP analysis -------------------

if (!dir.exists("figures")) dir.create("figures")
if (!dir.exists("tables")) dir.create("tables")

shap_vals <- SHAPforxgboost::shap.values(xgb_model = final_model, X_train = as.matrix(X))
shap_long <- SHAPforxgboost::shap.prep(shap_contrib = shap_vals$shap_score, X_train = as.data.frame(X))

imp_tbl <- data.frame(
  Feature = names(shap_vals$mean_shap_score),
  MeanAbsSHAP = as.numeric(shap_vals$mean_shap_score),
  row.names = NULL
) %>% arrange(desc(MeanAbsSHAP))

write_csv(imp_tbl, "tables/SHAP_global_importance.csv")
write_csv(imp_tbl[1:min(20, nrow(imp_tbl)), ], "tables/SHAP_global_importance_top20.csv")

# Beeswarm & bar plots
topN <- min(20, nrow(imp_tbl))
shap_long_top <- shap_long %>% filter(variable %in% imp_tbl$Feature[1:topN])

p_bee <- SHAPforxgboost::shap.plot.summary(shap_long_top) + 
  labs(
    title = "Distribution of SHAP values for top predictors",
    x = "SHAP value (impact on hospital admission risk)",
    y = "Predictor"
  )
ggsave("figures/SHAP_top20_beeswarm.png", p_bee, width = 8, height = 6, dpi = 300)

p_bar <- SHAPforxgboost::shap.plot.summary(shap_long_top, kind = "bar", dilute = TRUE) +
  labs(
    title = "Global feature importance (XGBoost model)",
    x = "Mean absolute SHAP value (average impact on admission prediction)",
    y = "Predictor"
  )
ggsave("figures/SHAP_top20_bar.png", p_bar, width = 7.5, height = 6, dpi = 300)

# Dependence plots for top 3 predictors
top_feats <- imp_tbl$Feature[1:min(3, nrow(imp_tbl))]

for (feat in top_feats) {
  p_dep <- SHAPforxgboost::shap.plot.dependence(shap_long, x = feat, smooth = FALSE)
  
  if (feat == "bmi") {
    p_dep <- p_dep + labs(
      title = "SHAP dependence plot for BMI",
      x = "Body mass index (kg/m²)",
      y = "SHAP value (impact on admission risk)"
    )
  } else if (feat == "Total_Yearly_Dose") {
    p_dep <- p_dep + labs(
      title = "SHAP dependence plot for annual furosemide dose",
      x = "Total yearly furosemide dose (mg/year)",
      y = "SHAP value (impact on admission risk)"
    )
  } else if (feat == "Year_sinceHF") {
    p_dep <- p_dep + labs(
      title = "SHAP dependence plot for years since heart failure diagnosis",
      x = "Years since heart failure diagnosis",
      y = "SHAP value (impact on admission risk)"
    )
  } else {
    p_dep <- p_dep + labs(
      title = paste0("SHAP dependence plot for ", feat),
      x = feat,
      y = "SHAP value (impact on admission risk)"
    )
  }
  
  ggsave(paste0("figures/SHAP_dependence_", feat, ".png"), p_dep, width = 7, height = 6, dpi = 300)
}

message("Done. See figures/ and tables/ for outputs.")

# ------------------- Save model -------------------

if (!dir.exists("models")) dir.create("models")
saveRDS(final_model, "models/final_xgb.rds")

predict_new_patients <- function(newdata, model_path = "models/final_xgb.rds") {
  model <- readRDS(model_path)
  X_new <- model.matrix(~ . - 1, data = newdata)
  pred_probs <- predict(model, newdata = X_new)
  data.frame(prob_yes = pred_probs, class = ifelse(pred_probs >= 0.5, "Yes", "No"))
}

cat("Model saved to models/final_xgb.rds. Use predict_new_patients() for new predictions.\n")









