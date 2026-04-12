# ============================================================
# Análise: Indian Liver Patient Dataset (ILPD)
# Estratégia: Superar resultados via XGBoost tunado, SMOTE,
#             feature engineering e ensemble com stacking
# Seed: 123 | Split: 80/20
# ============================================================

############################################################
# 0. INSTALAÇÃO DE PACOTES (EXECUTAR UMA VEZ SE NECESSÁRIO)
############################################################

# install_if_missing <- function(p) {
#   if (!require(p, character.only = TRUE)) {
#     install.packages(p)
#     library(p, character.only = TRUE)
#   }
# }
#
# lapply(c(
#   "dplyr", "ggplot2", "caret", "randomForest", "e1071",
#   "xgboost", "pROC", "tidyr", "MASS", "DMwR2", "gbm",
#   "glmnet", "kernlab", "MLmetrics", "tibble", "gridExtra"
# ), install_if_missing)
# install.packages("glmnet")
############################################################
# 1. PACOTES
############################################################

library(dplyr)
library(ggplot2)
library(caret)
library(randomForest)
library(e1071)
library(xgboost)
library(pROC)
library(tidyr)
library(MASS)
library(gbm)
library(glmnet)
library(tibble)
library(gridExtra)

# SMOTE — tenta DMwR2, cai em themis se precisar
if (!requireNamespace("DMwR2", quietly = TRUE) &&
    !requireNamespace("themis",  quietly = TRUE)) {
  stop("Instale DMwR2 ou themis para usar SMOTE.")
}

############################################################
# 2. CARREGAMENTO DOS DADOS
############################################################

path_base <- "Indian Liver Patient Dataset (ILPD).csv"

col_names <- c("Age", "Gender", "TB", "DB", "Alkphos",
               "SGPT", "SGOT", "TP", "ALB", "AG_Ratio", "Selector")

df <- read.csv(
  path_base,  # ← sem aspas aqui!
  header = FALSE,
  col.names = col_names,
  stringsAsFactors = FALSE
)

cat("Dimensões originais:", nrow(df), "x", ncol(df), "\n")
cat("Distribuição da variável alvo:\n")
print(table(df$Selector))

############################################################
# 3. PRÉ-PROCESSAMENTO
############################################################

# 3.1 Remover duplicatas exatas
df <- df[!duplicated(df), ]
cat("Após remoção de duplicatas:", nrow(df), "linhas\n")

# 3.2 Converter Gender para binário (Male=1, Female=0)
df$Gender <- ifelse(df$Gender == "Male", 1L, 0L)

# 3.3 Tratar valores ausentes em AG_Ratio (imputação pela mediana)
if (any(is.na(df$AG_Ratio))) {
  med_ag <- median(df$AG_Ratio, na.rm = TRUE)
  df$AG_Ratio[is.na(df$AG_Ratio)] <- med_ag
  cat("AG_Ratio: NAs imputados com mediana =", med_ag, "\n")
}

# 3.4 Converter alvo: 1=Liver Disease (Yes), 2=Healthy (No)
df$Selector <- ifelse(df$Selector == 1, "Yes", "No")
df$Selector <- factor(df$Selector, levels = c("No", "Yes"))

target <- "Selector"

############################################################
# 4. FEATURE ENGINEERING
############################################################
# Criar razões e interações biologicamente relevantes para o fígado

df <- df %>%
  mutate(
    # Razão TB/DB — indica tipo de hiperbilirrubinemia
    TB_DB_ratio    = ifelse(DB > 0, TB / DB, TB),
    # Razão SGPT/SGOT — indicador de lesão hepática
    SGPT_SGOT_ratio = ifelse(SGOT > 0, SGPT / SGOT, SGPT),
    # Bilirrubina indireta
    Indirect_Bili  = TB - DB,
    # Globulina (TP - ALB)
    Globulin       = TP - ALB,
    # Log das enzimas (reduz skewness)
    log_SGPT       = log1p(SGPT),
    log_SGOT       = log1p(SGOT),
    log_Alkphos    = log1p(Alkphos),
    log_TB         = log1p(TB),
    log_DB         = log1p(DB)
  )

cat("\nFeatures após engenharia:", ncol(df) - 1, "\n")

############################################################
# 5. SPLIT TREINO / TESTE (80 / 20, seed = 123)
############################################################

set.seed(123)
train_index <- createDataPartition(df[[target]], p = 0.80, list = FALSE)

train_raw <- df[train_index, ]
test_raw  <- df[-train_index, ]

cat("\nTreino:", nrow(train_raw), "| Teste:", nrow(test_raw), "\n")
cat("Distribuição treino:\n"); print(table(train_raw[[target]]))
cat("Distribuição teste:\n");  print(table(test_raw[[target]]))

############################################################
# 6. SELEÇÃO DE FEATURES + NORMALIZAÇÃO
############################################################

predictor_names <- setdiff(names(train_raw), target)

# 6.1 Remover near-zero variance
nzv_idx <- nearZeroVar(train_raw[, predictor_names])
if (length(nzv_idx) > 0) {
  predictor_names <- predictor_names[-nzv_idx]
  cat("Removidas por NZV:", length(nzv_idx), "variáveis\n")
}

# 6.2 Remover alta correlação (> 0.95) — mais conservador que 0.9
num_mask <- sapply(train_raw[, predictor_names], is.numeric)
if (sum(num_mask) > 1) {
  cor_mat  <- cor(train_raw[, predictor_names[num_mask]], use = "pairwise.complete.obs")
  high_cor <- findCorrelation(cor_mat, cutoff = 0.95)
  if (length(high_cor) > 0) {
    drop <- predictor_names[num_mask][high_cor]
    predictor_names <- setdiff(predictor_names, drop)
    cat("Removidas por alta correlação:", paste(drop, collapse = ", "), "\n")
  }
}

# 6.3 Normalização (center + scale)
pre_proc <- preProcess(
  train_raw[, predictor_names],
  method = c("center", "scale")
)

train_x <- predict(pre_proc, train_raw[, predictor_names])
test_x  <- predict(pre_proc, test_raw[, predictor_names])

train <- cbind(train_x, Selector = train_raw[[target]])
test  <- cbind(test_x,  Selector = test_raw[[target]])

predictor_names <- setdiff(names(train), target)

############################################################
# 7. BALANCEAMENTO DE CLASSES COM SMOTE
############################################################
# Aplicado SOMENTE no conjunto de treino

smote_train <- tryCatch({
  if (requireNamespace("DMwR2", quietly = TRUE)) {
    DMwR2::SMOTE(Selector ~ ., data = train, perc.over = 200, perc.under = 150)
  } else {
    # Fallback: themis via recipes — usa upSample simples
    caret::upSample(
      x = train[, predictor_names],
      y = train[[target]],
      yname = target
    )
  }
}, error = function(e) {
  cat("SMOTE falhou, usando upSample como fallback:", conditionMessage(e), "\n")
  caret::upSample(
    x = train[, predictor_names],
    y = train[[target]],
    yname = target
  )
})

# Garantir que o alvo ainda é fator com os níveis corretos
smote_train[[target]] <- factor(smote_train[[target]], levels = c("No", "Yes"))

cat("\nTreino após SMOTE:\n"); print(table(smote_train[[target]]))

############################################################
# 8. CONTROLE DE TREINAMENTO (CV Repetido - Otimiza AUC)
############################################################

train_ctrl <- trainControl(
  method          = "repeatedcv",
  number          = 10,
  repeats         = 3,
  classProbs      = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final",
  allowParallel   = TRUE
)

############################################################
# 9. SELEÇÃO DE VARIÁVEIS — StepAIC (backward)
############################################################

get_backward_vars <- function(train_df, target_name) {
  full_model <- glm(
    as.formula(paste(target_name, "~ .")),
    data   = train_df,
    family = binomial()
  )
  step_model <- stepAIC(full_model, direction = "backward", trace = FALSE)
  vars <- setdiff(names(coef(step_model)), "(Intercept)")
  unique(vars)
}

selected_vars <- tryCatch(
  get_backward_vars(smote_train, target),
  error = function(e) predictor_names
)
selected_vars <- intersect(selected_vars, predictor_names)
if (length(selected_vars) < 3) selected_vars <- predictor_names

cat("\nVariáveis selecionadas (StepAIC):", length(selected_vars), "\n")
cat(paste(selected_vars, collapse = ", "), "\n")

# Subconjuntos finais de treino/teste com variáveis selecionadas
train_sel <- smote_train[, c(target, selected_vars)]
test_sel  <- test[, c(target, selected_vars)]

############################################################
# 10. FUNÇÕES AUXILIARES
############################################################

f1_score <- function(cm) {
  p <- as.numeric(cm$byClass["Pos Pred Value"])
  r <- as.numeric(cm$byClass["Sensitivity"])
  if (is.na(p) || is.na(r) || (p + r) == 0) return(0)
  2 * (p * r) / (p + r)
}

optimize_threshold <- function(obs, prob) {
  roc_obj <- roc(obs, prob, quiet = TRUE)
  thr_df  <- coords(roc_obj, x = "best", best.method = "youden",
                    ret = "threshold", transpose = FALSE)
  thr <- as.numeric(thr_df[1, "threshold"])
  if (is.na(thr) || length(thr) == 0) thr <- 0.5
  thr
}

avaliar_prob <- function(prob, obs, thr = 0.5) {
  pred    <- factor(ifelse(prob > thr, "Yes", "No"), levels = c("No", "Yes"))
  cm      <- confusionMatrix(pred, obs, positive = "Yes")
  roc_obj <- roc(obs, prob, quiet = TRUE)
  list(cm = cm, auc = as.numeric(auc(roc_obj)), threshold = thr, prob = prob)
}

############################################################
# 11. MODELO 1 — RANDOM FOREST (tunado)
############################################################

cat("\n[1/5] Treinando Random Forest...\n")
set.seed(123)

rf_grid <- expand.grid(mtry = c(2, 3, 4, 5, 6))

model_rf <- train(
  as.formula(paste(target, "~ .")),
  data      = train_sel,
  method    = "rf",
  trControl = train_ctrl,
  metric    = "ROC",
  tuneGrid  = rf_grid,
  ntree     = 600
)
cat("Melhor mtry (RF):", model_rf$bestTune$mtry, "\n")

############################################################
# 12. MODELO 2 — XGBOOST DART (nativo)
############################################################
cat("\n[2/6] Treinando XGBoost DART...\n")
set.seed(123)

# Função auxiliar para extrair best_nrounds do log
get_best_nrounds <- function(cv_obj, fallback = 100L) {
  log     <- as.data.frame(cv_obj$evaluation_log)
  auc_col <- grep("test.*auc.*mean", names(log),
                  ignore.case = TRUE, value = TRUE)
  if (length(auc_col) > 0) {
    best <- which.max(log[[auc_col[1]]])
  } else {
    best <- nrow(log)
  }
  if (is.na(best) || length(best) == 0 || best < 1) best <- fallback
  as.integer(best)
}

x_train_xgb <- as.matrix(train_sel[, selected_vars])
y_train_xgb <- ifelse(train_sel[[target]] == "Yes", 1, 0)
x_test_xgb  <- as.matrix(test_sel[, selected_vars])

dtrain_xgb  <- xgb.DMatrix(data = x_train_xgb, label = y_train_xgb)

params_dart <- list(
  booster          = "dart",
  objective        = "binary:logistic",
  eval_metric      = "auc",
  eta              = 0.05,
  max_depth        = 6,
  subsample        = 0.8,
  colsample_bytree = 0.8,
  gamma            = 0,
  rate_drop        = 0.1,
  skip_drop        = 0.5
)

cv_dart   <- xgb.cv(
  data                  = dtrain_xgb,
  params                = params_dart,
  nrounds               = 500,
  nfold                 = 10,
  early_stopping_rounds = 25,
  verbose               = 0
)
best_dart <- get_best_nrounds(cv_dart)
cat("Melhor nrounds (dart):", best_dart, "\n")

model_dart <- xgb.train(
  params  = params_dart,
  data    = dtrain_xgb,
  nrounds = best_dart,
  verbose = 0
)

############################################################
# 13. MODELO 3 — XGBOOST GBLINEAR (nativo)
############################################################
cat("\n[3/6] Treinando XGBoost GBLINEAR...\n")
set.seed(123)

params_gblinear <- list(
  booster     = "gblinear",
  objective   = "binary:logistic",
  eval_metric = "auc",
  eta         = 0.05,
  lambda      = 0.01,
  alpha       = 0.01
)

cv_gblinear   <- xgb.cv(
  data                  = dtrain_xgb,
  params                = params_gblinear,
  nrounds               = 500,
  nfold                 = 10,
  early_stopping_rounds = 25,
  verbose               = 0
)
best_gblinear <- get_best_nrounds(cv_gblinear)
cat("Melhor nrounds (gblinear):", best_gblinear, "\n")

model_gblinear <- xgb.train(
  params  = params_gblinear,
  data    = dtrain_xgb,
  nrounds = best_gblinear,
  verbose = 0
)

############################################################
# 14. MODELO 4 — SVM RADIAL (caret)
############################################################
cat("\n[4/6] Treinando SVM Radial...\n")
set.seed(123)

model_svm <- train(
  as.formula(paste(target, "~ .")),
  data       = train_sel,
  method     = "svmRadial",
  trControl  = train_ctrl,
  metric     = "ROC",
  tuneLength = 12
)
cat("Melhor config SVM:\n"); print(model_svm$bestTune)

############################################################
# 15. MODELO 5 — GBM (caret)
############################################################
cat("\n[5/6] Treinando GBM...\n")
set.seed(123)

gbm_grid <- expand.grid(
  n.trees           = c(200, 300, 400),
  interaction.depth = c(3, 5),
  shrinkage         = c(0.05, 0.1),
  n.minobsinnode    = 10
)

model_gbm <- train(
  as.formula(paste(target, "~ .")),
  data      = train_sel,
  method    = "gbm",
  trControl = train_ctrl,
  metric    = "ROC",
  tuneGrid  = gbm_grid,
  verbose   = FALSE
)
cat("Melhor config GBM:\n"); print(model_gbm$bestTune)

############################################################
# 16. MODELO 6 — ELASTIC NET / REGRESSÃO LOGÍSTICA (caret)
############################################################
cat("\n[6/6] Treinando Elastic Net...\n")
set.seed(123)

glmnet_grid <- expand.grid(
  alpha  = c(0, 0.25, 0.5, 0.75, 1),
  lambda = 10^seq(-4, -1, length.out = 20)
)

model_glmnet <- train(
  as.formula(paste(target, "~ .")),
  data      = train_sel,
  method    = "glmnet",
  trControl = train_ctrl,
  metric    = "ROC",
  tuneGrid  = glmnet_grid,
  family    = "binomial"
)
cat("Melhor config Elastic Net:\n"); print(model_glmnet$bestTune)

############################################################
# 17. AVALIAÇÃO BASE (threshold = 0.5)
############################################################
cat("\n====== AVALIAÇÃO BASE (threshold = 0.50) ======\n")

# Modelos caret — predição padrão
prob_rf      <- predict(model_rf,      test_sel, type = "prob")[, "Yes"]
prob_svm     <- predict(model_svm,     test_sel, type = "prob")[, "Yes"]
prob_gbm     <- predict(model_gbm,     test_sel, type = "prob")[, "Yes"]
prob_glmnet  <- predict(model_glmnet,  test_sel, type = "prob")[, "Yes"]

# Modelos XGBoost nativos — predição direta
prob_dart    <- predict(model_dart,    x_test_xgb)
prob_gblinear <- predict(model_gblinear, x_test_xgb)

obs <- test_sel[[target]]

res_rf       <- avaliar_prob(prob_rf,       obs)
res_dart     <- avaliar_prob(prob_dart,     obs)
res_gblinear <- avaliar_prob(prob_gblinear, obs)
res_svm      <- avaliar_prob(prob_svm,      obs)
res_gbm      <- avaliar_prob(prob_gbm,      obs)
res_glmnet   <- avaliar_prob(prob_glmnet,   obs)

############################################################
# 18. ENSEMBLE — STACKING PONDERADO PELA AUC DO CV
############################################################

# AUC de CV: caret usa $results$ROC, xgboost usa o log
auc_dart_cv <- max(as.data.frame(cv_dart$evaluation_log)$test_auc_mean,
                   na.rm = TRUE)
auc_gbl_cv  <- max(as.data.frame(cv_gblinear$evaluation_log)$test_auc_mean,
                   na.rm = TRUE)

auc_cv <- c(
  RF       = max(model_rf$results$ROC),
  DART     = auc_dart_cv,
  GBLINEAR = auc_gbl_cv,
  SVM      = max(model_svm$results$ROC),
  GBM      = max(model_gbm$results$ROC),
  GLMNET   = max(model_glmnet$results$ROC)
)

cat("\nAUC de CV por modelo:\n"); print(round(auc_cv, 4))

weights   <- auc_cv / sum(auc_cv)

prob_stack <- (prob_rf       * weights["RF"]       +
                 prob_dart     * weights["DART"]     +
                 prob_gblinear * weights["GBLINEAR"] +
                 prob_svm      * weights["SVM"]      +
                 prob_gbm      * weights["GBM"]      +
                 prob_glmnet   * weights["GLMNET"])

res_stack <- avaliar_prob(prob_stack, obs)
cat("AUC Ensemble Stack:", round(res_stack$auc, 4), "\n")

############################################################
# 19. OTIMIZAÇÃO DE THRESHOLD (Índice de Youden)
############################################################

# Probabilidades no TREINO (sem data leakage)
prob_rf_tr       <- predict(model_rf,      train_sel, type = "prob")[, "Yes"]
prob_dart_tr     <- predict(model_dart,    x_train_xgb)
prob_gblinear_tr <- predict(model_gblinear, x_train_xgb)
prob_svm_tr      <- predict(model_svm,     train_sel, type = "prob")[, "Yes"]
prob_gbm_tr      <- predict(model_gbm,     train_sel, type = "prob")[, "Yes"]
prob_glmnet_tr   <- predict(model_glmnet,  train_sel, type = "prob")[, "Yes"]
obs_tr           <- train_sel[[target]]

prob_stack_tr <- (prob_rf_tr       * weights["RF"]       +
                    prob_dart_tr     * weights["DART"]     +
                    prob_gblinear_tr * weights["GBLINEAR"] +
                    prob_svm_tr      * weights["SVM"]      +
                    prob_gbm_tr      * weights["GBM"]      +
                    prob_glmnet_tr   * weights["GLMNET"])

thr_rf       <- optimize_threshold(obs_tr, prob_rf_tr)
thr_dart     <- optimize_threshold(obs_tr, prob_dart_tr)
thr_gblinear <- optimize_threshold(obs_tr, prob_gblinear_tr)
thr_svm      <- optimize_threshold(obs_tr, prob_svm_tr)
thr_gbm      <- optimize_threshold(obs_tr, prob_gbm_tr)
thr_glmnet   <- optimize_threshold(obs_tr, prob_glmnet_tr)
thr_stack    <- optimize_threshold(obs_tr, prob_stack_tr)

cat("\nThresholds otimizados (Youden):\n")
cat("RF:", round(thr_rf, 3),
    "| DART:", round(thr_dart, 3),
    "| GBLINEAR:", round(thr_gblinear, 3),
    "| SVM:", round(thr_svm, 3),
    "| GBM:", round(thr_gbm, 3),
    "| ElasticNet:", round(thr_glmnet, 3),
    "| Stack:", round(thr_stack, 3), "\n")

res_rf_thr       <- avaliar_prob(prob_rf,       obs, thr_rf)
res_dart_thr     <- avaliar_prob(prob_dart,     obs, thr_dart)
res_gblinear_thr <- avaliar_prob(prob_gblinear, obs, thr_gblinear)
res_svm_thr      <- avaliar_prob(prob_svm,      obs, thr_svm)
res_gbm_thr      <- avaliar_prob(prob_gbm,      obs, thr_gbm)
res_glmnet_thr   <- avaliar_prob(prob_glmnet,   obs, thr_glmnet)
res_stack_thr    <- avaliar_prob(prob_stack,    obs, thr_stack)

############################################################
# 20. TABELAS DE RESULTADOS
############################################################

extrair_metricas <- function(res, nome) {
  data.frame(
    Modelo      = nome,
    Threshold   = round(res$threshold, 3),
    Accuracy    = round(as.numeric(res$cm$overall["Accuracy"]),      3),
    AUC         = round(res$auc,                                      3),
    Sensitivity = round(as.numeric(res$cm$byClass["Sensitivity"]),    3),
    Specificity = round(as.numeric(res$cm$byClass["Specificity"]),    3),
    Precision   = round(as.numeric(res$cm$byClass["Pos Pred Value"]), 3),
    F1          = round(f1_score(res$cm),                             3),
    stringsAsFactors = FALSE
  )
}

# Tabela 1: baseline (thr = 0.5)
tab_base <- bind_rows(
  extrair_metricas(res_rf,       "Random Forest"),
  extrair_metricas(res_dart,     "XGB DART"),
  extrair_metricas(res_gblinear, "XGB GBLinear"),
  extrair_metricas(res_svm,      "SVM Radial"),
  extrair_metricas(res_gbm,      "GBM"),
  extrair_metricas(res_glmnet,   "Elastic Net (LR)"),
  extrair_metricas(res_stack,    "Ensemble Stack")
) %>% arrange(desc(AUC), desc(Accuracy))

# Tabela 2: threshold otimizado
tab_thr <- bind_rows(
  extrair_metricas(res_rf_thr,       "Random Forest (thr opt.)"),
  extrair_metricas(res_dart_thr,     "XGB DART (thr opt.)"),
  extrair_metricas(res_gblinear_thr, "XGB GBLinear (thr opt.)"),
  extrair_metricas(res_svm_thr,      "SVM Radial (thr opt.)"),
  extrair_metricas(res_gbm_thr,      "GBM (thr opt.)"),
  extrair_metricas(res_glmnet_thr,   "Elastic Net (thr opt.)"),
  extrair_metricas(res_stack_thr,    "Ensemble Stack (thr opt.)")
) %>% arrange(desc(AUC), desc(Accuracy))

cat("\n=== RESULTADOS BASE (threshold = 0.50) ===\n")
print(tab_base)

cat("\n=== RESULTADOS COM THRESHOLD OTIMIZADO ===\n")
print(tab_thr)

############################################################
# 21. IDENTIFICAR MELHOR MODELO GERAL
############################################################

todos <- bind_rows(tab_base, tab_thr) %>%
  arrange(desc(AUC), desc(Accuracy), desc(F1))

best_name <- todos$Modelo[1]
cat("\n>>> MELHOR MODELO:", best_name, "<<<\n")
cat("AUC:", todos$AUC[1],
    "| Acc:", todos$Accuracy[1],
    "| Sensitivity:", todos$Sensitivity[1],
    "| Specificity:", todos$Specificity[1],
    "| F1:", todos$F1[1], "\n")

# Selecionar probabilidades e threshold do melhor modelo
best_prob <- switch(
  gsub(" \\(thr opt\\.\\)", "", best_name),
  "Random Forest"    = prob_rf,
  "XGB DART"         = prob_dart,
  "XGB GBLinear"     = prob_gblinear,
  "SVM Radial"       = prob_svm,
  "GBM"              = prob_gbm,
  "Elastic Net (LR)" = prob_glmnet,
  "Ensemble Stack"   = prob_stack,
  prob_stack  # fallback
)
best_thr  <- todos$Threshold[1]

best_pred <- factor(ifelse(best_prob > best_thr, "Yes", "No"), levels = c("No", "Yes"))
cm_best   <- confusionMatrix(best_pred, obs, positive = "Yes")
roc_best  <- roc(obs, best_prob, quiet = TRUE)

############################################################
# 22. GRÁFICOS
############################################################

# ── 22.1 Comparativo Base ──
metricas_long <- tab_base %>%
  dplyr::select(Modelo, Accuracy, AUC, F1) %>%
  pivot_longer(cols = c(Accuracy, AUC, F1),
               names_to = "Metrica", values_to = "Valor")

p_base_chart <- ggplot(metricas_long,
                       aes(x = reorder(Modelo, Valor), y = Valor, fill = Metrica)) +
  geom_col(position = position_dodge(0.8), width = 0.7) +
  scale_y_continuous(limits = c(0, 1)) +
  coord_flip() +
  theme_minimal(base_size = 11) +
  labs(title = "Comparação Base dos Algoritmos (thr = 0.50)",
       x = NULL, y = "Score")

# ── 22.2 Comparativo Threshold Otimizado ──
metricas_thr_long <- tab_thr %>%
  mutate(Modelo = gsub(" \\(thr opt\\.\\)", "", Modelo)) %>%
  dplyr::select(Modelo, Accuracy, AUC, F1) %>%
  pivot_longer(cols = c(Accuracy, AUC, F1),
               names_to = "Metrica", values_to = "Valor")

p_thr_chart <- ggplot(metricas_thr_long,
                      aes(x = reorder(Modelo, Valor), y = Valor, fill = Metrica)) +
  geom_col(position = position_dodge(0.8), width = 0.7) +
  scale_y_continuous(limits = c(0, 1)) +
  coord_flip() +
  theme_minimal(base_size = 11) +
  labs(title = "Comparação com Threshold Otimizado (Youden)",
       x = NULL, y = "Score")

print(p_base_chart)
print(p_thr_chart)

# ── 22.3 Curva ROC do Melhor Modelo ──
roc_df <- data.frame(
  FPR = 1 - roc_best$specificities,
  TPR = roc_best$sensitivities
)

p_roc <- ggplot(roc_df, aes(x = FPR, y = TPR)) +
  geom_line(color = "firebrick", linewidth = 1.2) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray60") +
  theme_minimal(base_size = 12) +
  labs(
    title    = paste0("Curva ROC — ", best_name),
    subtitle = paste0("AUC = ", round(as.numeric(auc(roc_best)), 4)),
    x = "1 - Especificidade (FPR)",
    y = "Sensibilidade (TPR)"
  )
print(p_roc)

# ── 22.4 Matriz de Confusão ──
cm_df <- as.data.frame(cm_best$table)
colnames(cm_df) <- c("Predito", "Real", "Freq")

p_cm <- ggplot(cm_df, aes(x = Predito, y = Real, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), color = "white", size = 6, fontface = "bold") +
  scale_fill_gradient(low = "#5DA5DA", high = "#0B3C5D") +
  theme_minimal(base_size = 12) +
  labs(
    title    = paste0("Matriz de Confusão — ", best_name),
    subtitle = paste0("Threshold = ", round(best_thr, 3))
  )
print(p_cm)

# ── 22.5 Importância das Variáveis (Random Forest) ──
imp_df <- varImp(model_rf, scale = TRUE)$importance %>%
  rownames_to_column("Variavel") %>%
  mutate(Importancia = rowMeans(dplyr::select(., -Variavel))) %>%
  dplyr::select(Variavel, Importancia) %>%
  arrange(desc(Importancia)) %>%
  slice_head(n = 15)

p_imp <- ggplot(imp_df, aes(x = reorder(Variavel, Importancia), y = Importancia)) +
  geom_col(fill = "#2E8B57", width = 0.8) +
  coord_flip() +
  theme_minimal(base_size = 11) +
  labs(
    title    = "Importância das Variáveis — Random Forest",
    subtitle = "Top 15 por importância média (escala 0-100)",
    x = NULL, y = "Importância"
  )
print(p_imp)

# ── 22.6 Importância das Variáveis (XGB DART) ──
imp_dart <- xgb.importance(model = model_dart) %>%
  as.data.frame() %>%
  slice_head(n = 15)

p_imp_dart <- ggplot(imp_dart,
                     aes(x = reorder(Feature, Gain), y = Gain)) +
  geom_col(fill = "#E07B39", width = 0.8) +
  coord_flip() +
  theme_minimal(base_size = 11) +
  labs(
    title    = "Importância das Variáveis — XGB DART",
    subtitle = "Top 15 por Gain",
    x = NULL, y = "Gain"
  )
print(p_imp_dart)

# ── 22.7 Distribuição de Probabilidades ──
prob_df <- data.frame(Prob = best_prob, Classe = obs)

p_dist <- ggplot(prob_df, aes(x = Prob, fill = Classe)) +
  geom_histogram(position = "identity", alpha = 0.55, bins = 25) +
  geom_vline(xintercept = best_thr, linetype = "dashed",
             color = "red", linewidth = 1) +
  theme_minimal(base_size = 11) +
  labs(
    title    = paste0("Distribuição de Probabilidades — ", best_name),
    subtitle = "Linha vermelha = threshold otimizado",
    x = "P(Liver Disease)", y = "Contagem"
  )
print(p_dist)

# ── 22.8 Impacto do Limiar ──
thr_grid <- seq(0.05, 0.95, by = 0.01)
thr_impact <- lapply(thr_grid, function(thr) {
  p  <- factor(ifelse(best_prob > thr, "Yes", "No"), levels = c("No", "Yes"))
  cm <- confusionMatrix(p, obs, positive = "Yes")
  data.frame(
    Threshold   = thr,
    Accuracy    = as.numeric(cm$overall["Accuracy"]),
    Sensitivity = as.numeric(cm$byClass["Sensitivity"]),
    Specificity = as.numeric(cm$byClass["Specificity"]),
    F1          = f1_score(cm)
  )
}) %>% bind_rows() %>%
  pivot_longer(c(Accuracy, Sensitivity, Specificity, F1),
               names_to = "Metrica", values_to = "Valor")

p_thr_impact <- ggplot(thr_impact,
                       aes(x = Threshold, y = Valor, color = Metrica)) +
  geom_line(linewidth = 1) +
  geom_vline(xintercept = best_thr, linetype = "dashed",
             color = "black", linewidth = 0.8) +
  scale_y_continuous(limits = c(0, 1)) +
  theme_minimal(base_size = 11) +
  labs(
    title    = paste0("Impacto do Limiar — ", best_name),
    subtitle = paste0("Limiar otimizado = ", round(best_thr, 3)),
    x = "Threshold", y = "Score", color = "Métrica"
  )
print(p_thr_impact)

############################################################
# 23. COMPARAÇÃO COM COLEGAS
############################################################

colegas <- data.frame(
  Fonte  = "Colegas (artigo)",
  Modelo = c("Regressão Logística", "GLM", "SVM", "Random Forest",
             "SVM Class Weights", "ANN", "Gaussian NB"),
  ACC_SemCFS = c(75.3, 73.0, 70.7, 70.7, 71.3, 69.5, 59.8)
)

cat("\n=== REFERÊNCIA DOS COLEGAS (Acurácia sem CFS, base completa) ===\n")
print(colegas)

cat("\n=== COMPARATIVO NOSSO vs COLEGAS ===\n")
cat("Melhor dos colegas: RL = 75.3%\n")
cat(sprintf("Nosso melhor (base):     %.1f%%\n", max(tab_base$Accuracy) * 100))
cat(sprintf("Nosso melhor (thr opt.): %.1f%%\n", max(tab_thr$Accuracy)  * 100))
cat(sprintf("Nosso melhor AUC:        %.4f\n",   max(tab_thr$AUC)))

############################################################
# 24. RESULTADO FINAL CONSOLIDADO
############################################################

cat("\n")
cat("=======================================================\n")
cat("        RESULTADOS FINAIS — ILPD (seed=123, 80/20)     \n")
cat("=======================================================\n")
cat("\n--- Baseline (threshold = 0.50) ---\n")
print(tab_base)
cat("\n--- Com Threshold Otimizado (Youden) ---\n")
print(tab_thr)
cat("\n--- Melhor Modelo ---\n")
cat("Modelo:", best_name, "\n")
cat("AUC:", todos$AUC[1],
    "| Accuracy:", todos$Accuracy[1],
    "| Sensitivity:", todos$Sensitivity[1],
    "| Specificity:", todos$Specificity[1],
    "| F1:", todos$F1[1], "\n")
cat("=======================================================\n")