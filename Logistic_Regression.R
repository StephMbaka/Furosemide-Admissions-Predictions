library(dplyr)
library(broom)
library(tidyr)
library(readxl)
library(openxlsx)
library(pROC)
library(openxlsx)

# ---------------- Load & prep ----------------
df <- read_excel("completedata_NoNAs.xlsx") %>%
  filter(Year_sinceHF %in% 0:10) %>%
  mutate(
    # Use your existing age variable if present
    Age = if ("age_at_event" %in% names(.)) age_at_event else Age,
    admitted = ifelse(admitted == "Yes", 1L, 0L),
    # Tidy categorical variables
    smoking = replace_na(as.character(smoking), "Other"),
    alcohol = replace_na(as.character(alcohol), "Other"),
    sex = factor(sex),
    smoking = factor(smoking),
    alcohol = factor(alcohol)
  )

# ---------------- Per-year fits ----------------
years <- sort(unique(df$Year_sinceHF))
coef_list   <- list()
metric_list <- list()

for (y in years) {
  dat <- df %>%
    filter(Year_sinceHF == y) %>%
    drop_na(admitted, Total_Yearly_Dose, sex, smoking)
  
  # Need both classes
  if (dplyr::n_distinct(dat$admitted) < 2) {
    message(sprintf("Skipping Year %s (only one outcome class).", y))
    next
  }
  
  # Fit: adjust predictors as you wish (can add bmi, Age, NumberofComobidites, alcohol, etc.)
  fit <- glm(admitted ~ Total_Yearly_Dose + sex + smoking,
             data = dat, family = binomial)
  
  # ---- Coefficients as ORs with 95% CI ----
  coefs <- tidy(fit, exponentiate = TRUE, conf.int = TRUE) %>%
    mutate(Year = y) %>%
    select(Year, term, estimate, conf.low, conf.high, p.value)
  coef_list[[as.character(y)]] <- coefs
  
  # ---- Predictions & AUC ----
  pr <- predict(fit, type = "response")
  roc_obj <- pROC::roc(dat$admitted, pr, direction = "<", quiet = TRUE)
  auc_val <- as.numeric(pROC::auc(roc_obj))
  auc_ci  <- pROC::ci.auc(roc_obj)
  
  # Optimal threshold by Youden's J
  youden <- pROC::coords(roc_obj, x = "best", best.method = "youden",
                         ret = c("threshold", "sensitivity", "specificity"))
  
  thr <- as.numeric(youden["threshold"])
  sens <- as.numeric(youden["sensitivity"])
  spec <- as.numeric(youden["specificity"])
  
  # Confusion metrics at optimal threshold
  pred_cls <- ifelse(pr >= thr, 1L, 0L)
  TP <- sum(pred_cls == 1 & dat$admitted == 1)
  TN <- sum(pred_cls == 0 & dat$admitted == 0)
  FP <- sum(pred_cls == 1 & dat$admitted == 0)
  FN <- sum(pred_cls == 0 & dat$admitted == 1)
  acc <- (TP + TN) / (TP + TN + FP + FN)
  
  # Model info
  metrics <- tibble(
    Year = y,
    N = nrow(dat),
    Positives = sum(dat$admitted == 1),
    Negatives = sum(dat$admitted == 0),
    AUC = auc_val,
    AUC_LCL = as.numeric(auc_ci[1]),
    AUC_UCL = as.numeric(auc_ci[3]),
    Threshold_Youden = thr,
    Sensitivity_Youden = sens,
    Specificity_Youden = spec,
    Accuracy_Youden = acc,
    AIC = AIC(fit),
    Null_Deviance = fit$null.deviance,
    Residual_Deviance = fit$deviance,
    DF_Null = fit$df.null,
    DF_Residual = fit$df.residual
  )
  
  metric_list[[as.character(y)]] <- metrics
}

coef_tbl   <- bind_rows(coef_list)
metrics_tbl <- bind_rows(metric_list) %>% arrange(Year)

# ---------------- Save outputs ----------------
write.csv(coef_tbl, "Logistic_by_year_coefficients.csv", row.names = FALSE)
write.csv(metrics_tbl, "Logistic_by_year_metrics.csv", row.names = FALSE)

wb <- createWorkbook()
addWorksheet(wb, "Coefficients")
addWorksheet(wb, "Metrics")
writeData(wb, "Coefficients", coef_tbl)
writeData(wb, "Metrics", metrics_tbl)
saveWorkbook(wb, "Logistic_by_year_results.xlsx", overwrite = TRUE)

cat("Saved:\n - Logistic_by_year_results.xlsx (Coefficients + Metrics)\n - Logistic_by_year_coefficients.csv\n - Logistic_by_year_metrics.csv\n")

library(ggplot2)

# AUC over years with 95% CI ribbons
p_auc <- ggplot(metrics_tbl, aes(x = Year, y = AUC)) +
  geom_ribbon(aes(ymin = AUC_LCL, ymax = AUC_UCL), alpha = 0.15) +
  geom_line(size = 0.8) +
  geom_point(size = 2) +
  scale_x_continuous(breaks = sort(unique(metrics_tbl$Year))) +
  coord_cartesian(ylim = c(0.5, 1)) +
  labs(title = "Model Discrimination over Follow-up Years",
       x = "Years since heart failure diagnosis",
       y = "Area under the ROC curve (AUC) with 95% CI") +
  theme_minimal(base_size = 12)

print(p_auc)
ggsave("AUC_by_year.png", p_auc, width = 7, height = 4.5, dpi = 300)


# Keep only the dose term, coerce to numeric, and ensure proper ordering
dose_tbl <- coef_tbl %>%
  dplyr::filter(term == "Total_Yearly_Dose") %>%
  dplyr::mutate(
    estimate  = as.numeric(estimate),
    conf.low  = as.numeric(conf.low),
    conf.high = as.numeric(conf.high)
  ) %>%
  dplyr::arrange(Year)

# Forest plot (log scale on x so symmetric CIs; OR=1 reference line)
p_forest <- ggplot(dose_tbl,
                   aes(y = factor(Year), x = estimate, xmin = conf.low, 
                       xmax = conf.high)) +
  geom_vline(xintercept = 1, linetype = "dashed") +
  geom_errorbarh(height = 0.15) +
  geom_point(size = 2) +
  scale_x_continuous(trans = "log10") +
  labs(title = "Effect of Total Yearly Furosemide Dose on Admission Risk",
       x = "Odds ratio (log scale, per 1 mg increase in annual dose)",
       y = "Years since heart failure diagnosis") +
  theme_minimal(base_size = 12)

print(p_forest)
ggsave("Forest_OR_Total_Yearly_Dose_by_year.png", p_forest, width = 7, height = 5, dpi = 300)
