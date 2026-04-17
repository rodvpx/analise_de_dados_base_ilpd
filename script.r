# ============================================================
# Análise: Indian Liver Patient Dataset (ILPD)
# Estratégia: seleção padronizada de variáveis, avaliação treino/teste,
#             feature engineering e ensemble com stacking
#
# Melhorias em relação à versão anterior:
#   [FIX 1] Seleção de variáveis (StepAIC) feita ANTES da modelagem,
#           garantindo que o mesmo conjunto de variáveis seja usado
#           por todos os modelos.
#   [FIX 2] Avaliação reportada tanto no TREINO quanto no TESTE,
#           permitindo diagnóstico de overfitting em cada modelo.
#   [FIX 3] Justificativas claras para a seleção de variáveis,
#           pesos de classe e decisões metodológicas documentadas.
#   [NEW]   Regressão Logística pura adicionada (modelo dos colegas)
#           para comparação direta na mesma pipeline.
#   [NEW]   Regressão Logística treinada no conjunto original
#           para comparação direta com o artigo base.
#
# Seed: 123 | Split: 80/20
# ============================================================

############################################################
# 0. INSTALAÇÃO DE PACOTES (EXECUTAR UMA VEZ SE NECESSÁRIO)
############################################################

# install_if_missing <- function(p) {
#   if (!require(p, character.only = TRUE)) install.packages(p)
# }
# lapply(c("dplyr", "ggplot2", "caret", "randomForest", "e1071",
#          "xgboost", "pROC", "tidyr", "MASS", "DMwR2", "gbm",
#          "glmnet", "kernlab", "tibble", "gridExtra"), install_if_missing)

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

# Não usamos balanceamento sintético nesta versão; apenas o treino original e, quando aplicável,
# pesos de classe para reduzir viés em bases desbalanceadas.

############################################################
# 2. CARREGAMENTO DOS DADOS
############################################################

path_base <- "Indian Liver Patient Dataset (ILPD).csv"

col_names <- c("Age", "Gender", "TB", "DB", "Alkphos",
               "SGPT", "SGOT", "TP", "ALB", "AG_Ratio", "Selector")

df <- read.csv(path_base, header = FALSE, col.names = col_names,
               stringsAsFactors = FALSE)

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

# 3.3 Imputar AG_Ratio (5 valores ausentes) com mediana do dataset
#     Mediana é mais robusta que média diante dos outliers típicos de
#     exames laboratoriais com assimetria positiva.
if (any(is.na(df$AG_Ratio))) {
  med_ag <- median(df$AG_Ratio, na.rm = TRUE)
  df$AG_Ratio[is.na(df$AG_Ratio)] <- med_ag
  cat("AG_Ratio: NAs imputados com mediana =", med_ag, "\n")
}

# 3.4 Converter alvo: 1 = Doença Hepática (Yes), 2 = Saudável (No)
df$Selector <- ifelse(df$Selector == 1, "Yes", "No")
df$Selector <- factor(df$Selector, levels = c("No", "Yes"))

target <- "Selector"

############################################################
# 4. FEATURE ENGINEERING
############################################################
# Criamos variáveis derivadas com significado clínico estabelecido
# na literatura de hepatologia. Isso enriquece o espaço de atributos
# sem violar o processo de separação treino/teste, pois são
# transformações determinísticas das variáveis originais.

df <- df %>%
  mutate(
    # Razão TB/DB: diferencia hiperbilirrubinemia direta vs indireta
    TB_DB_ratio     = ifelse(DB > 0, TB / DB, TB),
    # Razão SGPT/SGOT: indica origem celular do dano (>2 sugere hepatite alcoólica)
    SGPT_SGOT_ratio = ifelse(SGOT > 0, SGPT / SGOT, SGPT),
    # Bilirrubina indireta: marcador de hemólise e disfunção hepática
    Indirect_Bili   = TB - DB,
    # Globulina: estimada como TP - ALB, indica resposta inflamatória
    Globulin        = TP - ALB,
    # Transformações log: reduzem a assimetria das enzimas (distribuição mais próxima da normal)
    log_SGPT        = log1p(SGPT),
    log_SGOT        = log1p(SGOT),
    log_Alkphos     = log1p(Alkphos),
    log_TB          = log1p(TB),
    log_DB          = log1p(DB)
  )

cat("\nFeatures após engenharia:", ncol(df) - 1, "\n")

############################################################
# 5. SPLIT TREINO / TESTE (80 / 20, seed = 123)
############################################################
# createDataPartition realiza estratificação pela variável alvo,
# preservando a proporção de classes (71% Yes / 29% No) em ambos
# os conjuntos. Isso é essencial para avaliação não enviesada.

set.seed(123)
train_index <- createDataPartition(df[[target]], p = 0.80, list = FALSE)

train_raw <- df[train_index, ]
test_raw  <- df[-train_index, ]

cat("\nTreino:", nrow(train_raw), "| Teste:", nrow(test_raw), "\n")
cat("Distribuição treino:\n"); print(table(train_raw[[target]]))
cat("Distribuição teste:\n");  print(table(test_raw[[target]]))

############################################################
# 6. NORMALIZAÇÃO (center + scale)
############################################################
# O pré-processamento de escala é ajustado SOMENTE no treino e
# depois aplicado ao teste. Isso evita data leakage: o modelo de
# normalização não "vê" o conjunto de teste antes do momento de avaliação.

predictor_names <- setdiff(names(train_raw), target)

# 6.1 Remover near-zero variance (variáveis sem poder discriminativo)
nzv_idx <- nearZeroVar(train_raw[, predictor_names])
if (length(nzv_idx) > 0) {
  cat("Removidas por NZV:", length(nzv_idx), "variáveis\n")
  predictor_names <- predictor_names[-nzv_idx]
}

# 6.2 Remover alta correlação (> 0.95)
#     Correlação extrema indica redundância; manter apenas uma das variáveis
#     preserva a informação sem inflar a dimensionalidade.
num_mask <- sapply(train_raw[, predictor_names], is.numeric)
if (sum(num_mask) > 1) {
  cor_mat  <- cor(train_raw[, predictor_names[num_mask]], use = "pairwise.complete.obs")
  high_cor <- findCorrelation(cor_mat, cutoff = 0.95)
  if (length(high_cor) > 0) {
    drop <- predictor_names[num_mask][high_cor]
    predictor_names <- setdiff(predictor_names, drop)
    cat("Removidas por alta correlação (>0.95):", paste(drop, collapse = ", "), "\n")
  }
}

# 6.3 Normalizar: ajuste no treino, aplica em treino E teste
pre_proc <- preProcess(train_raw[, predictor_names], method = c("center", "scale"))

train_x <- predict(pre_proc, train_raw[, predictor_names])
test_x  <- predict(pre_proc, test_raw[, predictor_names])

train_norm <- cbind(train_x, Selector = train_raw[[target]])
test_norm  <- cbind(test_x,  Selector = test_raw[[target]])

predictor_names <- setdiff(names(train_norm), target)

############################################################
# 7. SELEÇÃO DE VARIÁVEIS — StepAIC backward
############################################################
# [FIX 1] A seleção é realizada sobre o conjunto de TREINO ORIGINAL
# (normalizado, sem balanceamento), de modo que:
#   a) O mesmo subconjunto de variáveis é usado por TODOS os modelos.
#   b) O balanceamento artificial não influencia quais variáveis são selecionadas
#      (evita viés de seleção introduzido por dados sintéticos).
#
# Por que StepAIC e não CFS?
# CFS foi desenvolvido para algoritmos baseados em filtros (Naïve Bayes,
# kNN) e tende a remover interações úteis para modelos modernos como
# XGBoost e Random Forest. StepAIC seleciona com base em critério de
# informação (AIC), preservando relações lineares relevantes e sendo
# mais robusto para pipelines com ensemble.

cat("\n[Seleção de Variáveis] Executando StepAIC backward no treino original...\n")

get_backward_vars <- function(train_df, target_name) {
  full_model  <- glm(as.formula(paste(target_name, "~ .")),
                     data = train_df, family = binomial())
  step_model  <- stepAIC(full_model, direction = "backward", trace = FALSE)
  vars <- setdiff(names(coef(step_model)), "(Intercept)")
  unique(vars)
}

selected_vars <- tryCatch(
  get_backward_vars(train_norm, target),
  error = function(e) {
    cat("StepAIC falhou, usando todas as variáveis:", conditionMessage(e), "\n")
    predictor_names
  }
)
selected_vars <- intersect(selected_vars, predictor_names)
if (length(selected_vars) < 3) selected_vars <- predictor_names

cat("Variáveis selecionadas (StepAIC):", length(selected_vars), "\n")
cat(paste(selected_vars, collapse = ", "), "\n")

# Aplicar seleção ao treino e ao teste
train_sel <- train_norm[, c(target, selected_vars)]
test_sel  <- test_norm[,  c(target, selected_vars)]

############################################################
# 8. TRATAMENTO DO DESBALANCEAMENTO SEM AMOSTRAGEM SINTÉTICA
############################################################
# Em vez de gerar amostras sintéticas, esta versão usa o conjunto original
# do treino e aplica pesos de classe apenas quando o método aceitar.
# Assim evitamos a distorção que o balanceamento artificial pode causar em modelos de árvore
# e mantemos a avaliação mais próxima do cenário real.

cat("\n[Desbalanceamento] Usando treino original (sem amostragem sintética).\n")
cat("Distribuição no treino original:\n"); print(table(train_sel[[target]]))

class_tab <- table(train_sel[[target]])
class_weights <- as.numeric(sum(class_tab) / (length(class_tab) * class_tab))
names(class_weights) <- names(class_tab)

obs_weights_train <- ifelse(train_sel[[target]] == "No",
                            class_weights["No"],
                            class_weights["Yes"])

cat("Pesos de classe calculados:\n")
print(round(class_weights, 4))

############################################################
# 9. CONTROLE DE TREINAMENTO (Repeated CV — otimiza AUC)
############################################################
# 10 folds × 3 repetições = 30 estimativas de generalização por modelo.
# Métrica alvo: ROC (AUC), que é mais informativa que accuracy em dados
# desbalanceados por considerar o tradeoff sensibilidade/especificidade.

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

# [FIX 2] Função para extrair métricas em TREINO e TESTE separadamente
extrair_metricas_split <- function(prob_tr, obs_tr, prob_te, obs_te, nome, thr = 0.5) {
  res_tr <- avaliar_prob(prob_tr, obs_tr, thr)
  res_te <- avaliar_prob(prob_te, obs_te, thr)
  data.frame(
    Modelo       = nome,
    Threshold    = round(thr, 3),
    # Métricas no TREINO
    Train_Acc    = round(as.numeric(res_tr$cm$overall["Accuracy"]),       3),
    Train_AUC    = round(res_tr$auc,                                       3),
    Train_Sens   = round(as.numeric(res_tr$cm$byClass["Sensitivity"]),    3),
    Train_Spec   = round(as.numeric(res_tr$cm$byClass["Specificity"]),    3),
    Train_F1     = round(f1_score(res_tr$cm),                             3),
    # Métricas no TESTE
    Test_Acc     = round(as.numeric(res_te$cm$overall["Accuracy"]),       3),
    Test_AUC     = round(res_te$auc,                                       3),
    Test_Sens    = round(as.numeric(res_te$cm$byClass["Sensitivity"]),    3),
    Test_Spec    = round(as.numeric(res_te$cm$byClass["Specificity"]),    3),
    Test_F1      = round(f1_score(res_te$cm),                             3),
    stringsAsFactors = FALSE
  )
}

############################################################
# 11. MODELO 1 — RANDOM FOREST (tunado)
############################################################

cat("\n[1/7] Treinando Random Forest...\n")
set.seed(123)

rf_grid  <- expand.grid(mtry = c(2, 3, 4, 5, 6))
model_rf <- train(
  as.formula(paste(target, "~ .")),
  data      = train_sel,
  method    = "rf",
  trControl = train_ctrl,
  metric    = "ROC",
  tuneGrid  = rf_grid,
  ntree     = 600,
  classwt   = class_weights
)
cat("Melhor mtry (RF):", model_rf$bestTune$mtry, "\n")
cat("AUC CV (RF):     ", round(max(model_rf$results$ROC), 4), "\n")

############################################################
# 12. MODELO 2 — XGBOOST DART (nativo)
############################################################
# O XGBoost é treinado diretamente pela biblioteca oficial e não via
# caret, pois o wrapper caret/xgbTree está em modo de manutenção e
# apresenta incompatibilidades com versões recentes do XGBoost.
# Isso nos permite usar:
#   • xgb.cv() com early stopping real (evita overfitting sem fixar nrounds)
#   • Booster DART (Dropouts meet Multiple Additive Regression Trees),
#     que aplica dropout em árvores para regularização adicional.

cat("\n[2/7] Treinando XGBoost DART...\n")
set.seed(123)

get_best_nrounds <- function(cv_obj, fallback = 100L) {
  log     <- as.data.frame(cv_obj$evaluation_log)
  auc_col <- grep("test.*auc.*mean", names(log), ignore.case = TRUE, value = TRUE)
  best    <- if (length(auc_col) > 0) which.max(log[[auc_col[1]]]) else nrow(log)
  if (is.na(best) || length(best) == 0 || best < 1) best <- fallback
  as.integer(best)
}

x_train_xgb <- as.matrix(train_sel[, selected_vars])
y_train_xgb <- ifelse(train_sel[[target]] == "Yes", 1, 0)
x_test_xgb  <- as.matrix(test_sel[, selected_vars])
# Para avaliação no treino com threshold (usamos train_sel original, sem amostragem sintética)
x_trainorig_xgb <- as.matrix(train_sel[, selected_vars])

dtrain_xgb <- xgb.DMatrix(data = x_train_xgb, label = y_train_xgb, weight = obs_weights_train)

params_dart <- list(
  booster = "dart", objective = "binary:logistic", eval_metric = "auc",
  eta = 0.05, max_depth = 6, subsample = 0.8, colsample_bytree = 0.8,
  gamma = 0, rate_drop = 0.1, skip_drop = 0.5
)

cv_dart   <- xgb.cv(data = dtrain_xgb, params = params_dart, nrounds = 500,
                    nfold = 10, early_stopping_rounds = 25, verbose = 0)
best_dart <- get_best_nrounds(cv_dart)
cat("Melhor nrounds (DART):", best_dart, "\n")

model_dart <- xgb.train(params = params_dart, data = dtrain_xgb,
                        nrounds = best_dart, verbose = 0)

############################################################
# 13. MODELO 3 — XGBOOST GBLINEAR (nativo)
############################################################

cat("\n[3/7] Treinando XGBoost GBLinear...\n")
set.seed(123)

params_gblinear <- list(
  booster = "gblinear", objective = "binary:logistic", eval_metric = "auc",
  eta = 0.05, lambda = 0.01, alpha = 0.01
)

cv_gblinear   <- xgb.cv(data = dtrain_xgb, params = params_gblinear, nrounds = 500,
                        nfold = 10, early_stopping_rounds = 25, verbose = 0)
best_gblinear <- get_best_nrounds(cv_gblinear)
cat("Melhor nrounds (GBLinear):", best_gblinear, "\n")

model_gblinear <- xgb.train(params = params_gblinear, data = dtrain_xgb,
                            nrounds = best_gblinear, verbose = 0)

############################################################
# 14. MODELO 4 — SVM RADIAL (caret)
############################################################

cat("\n[4/7] Treinando SVM Radial...\n")
set.seed(123)

model_svm <- train(
  as.formula(paste(target, "~ .")),
  data       = train_sel,
  method     = "svmRadial",
  trControl  = train_ctrl,
  metric     = "ROC",
  tuneLength = 12,
  class.weights = class_weights
)
cat("Melhor config SVM:\n"); print(model_svm$bestTune)

############################################################
# 15. MODELO 5 — GBM (caret)
############################################################

cat("\n[5/7] Treinando GBM...\n")
set.seed(123)

gbm_grid <- expand.grid(
  n.trees = c(200, 300, 400), interaction.depth = c(3, 5),
  shrinkage = c(0.05, 0.1), n.minobsinnode = 10
)

model_gbm <- train(
  as.formula(paste(target, "~ .")),
  data      = train_sel,
  method    = "gbm",
  trControl = train_ctrl,
  metric    = "ROC",
  tuneGrid  = gbm_grid,
  verbose   = FALSE,
  weights   = obs_weights_train
)
cat("Melhor config GBM:\n"); print(model_gbm$bestTune)

############################################################
# 16. MODELO 6 — ELASTIC NET / LOGÍSTICA REGULARIZADA (caret)
############################################################

cat("\n[6/7] Treinando Elastic Net...\n")
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
  family    = "binomial",
  weights   = obs_weights_train
)
cat("Melhor config Elastic Net:\n"); print(model_glmnet$bestTune)

############################################################
# 17. MODELO 7 — REGRESSÃO LOGÍSTICA PURA (sem regularização)
############################################################
# Regressão logística simples no conjunto original de treino.
# Ela funciona como referência direta para comparação com o artigo
# dos colegas e também serve como baseline interpretável.

cat("\n[7/7] Treinando Regressão Logística pura...\n")
set.seed(123)

model_lr_nosmt <- train(
  as.formula(paste(target, "~ .")),
  data      = train_sel,
  method    = "glm",
  family    = "binomial",
  trControl = train_ctrl,
  metric    = "ROC",
  weights   = obs_weights_train
)
cat("AUC CV (Regressão Logística):", round(max(model_lr_nosmt$results$ROC), 4), "\n")

############################################################
# 18. PREDIÇÕES (TREINO E TESTE)
############################################################
# [FIX 2] Cada modelo é avaliado tanto no conjunto de TREINO quanto
# no de TESTE. A diferença entre as métricas de treino e teste indica
# o grau de overfitting:
#   • Treino muito superior ao teste → modelo decorou os dados (overfitting)
#   • Treino e teste próximos      → boa generalização

obs_te <- test_sel[[target]]
obs_tr <- train_sel[[target]]   # treino ORIGINAL (sem amostragem sintética) para avaliação

# Modelos caret — treino e teste
prob_rf_te      <- predict(model_rf,       test_sel,  type = "prob")[, "Yes"]
prob_svm_te     <- predict(model_svm,      test_sel,  type = "prob")[, "Yes"]
prob_gbm_te     <- predict(model_gbm,      test_sel,  type = "prob")[, "Yes"]
prob_glmnet_te  <- predict(model_glmnet,   test_sel,  type = "prob")[, "Yes"]
prob_lr_no_te   <- predict(model_lr_nosmt, test_sel,  type = "prob")[, "Yes"]

prob_rf_tr      <- predict(model_rf,       train_sel, type = "prob")[, "Yes"]
prob_svm_tr     <- predict(model_svm,      train_sel, type = "prob")[, "Yes"]
prob_gbm_tr     <- predict(model_gbm,      train_sel, type = "prob")[, "Yes"]
prob_glmnet_tr  <- predict(model_glmnet,   train_sel, type = "prob")[, "Yes"]
prob_lr_no_tr   <- predict(model_lr_nosmt, train_sel, type = "prob")[, "Yes"]

# XGBoost — treino (train_sel original) e teste
prob_dart_te      <- predict(model_dart,    xgb.DMatrix(x_test_xgb))
prob_gblinear_te  <- predict(model_gblinear,xgb.DMatrix(x_test_xgb))
prob_dart_tr      <- predict(model_dart,    xgb.DMatrix(x_trainorig_xgb))
prob_gblinear_tr  <- predict(model_gblinear,xgb.DMatrix(x_trainorig_xgb))

############################################################
# 19. ENSEMBLE STACKING (ponderado por AUC de CV)
############################################################

auc_dart_cv <- max(as.data.frame(cv_dart$evaluation_log)$test_auc_mean,    na.rm = TRUE)
auc_gbl_cv  <- max(as.data.frame(cv_gblinear$evaluation_log)$test_auc_mean, na.rm = TRUE)

auc_cv <- c(
  RF       = max(model_rf$results$ROC),
  DART     = auc_dart_cv,
  GBLINEAR = auc_gbl_cv,
  SVM      = max(model_svm$results$ROC),
  GBM      = max(model_gbm$results$ROC),
  GLMNET   = max(model_glmnet$results$ROC),
  LR       = max(model_lr_nosmt$results$ROC)
)

cat("\nAUC de CV por modelo (usados como pesos do ensemble):\n")
print(round(auc_cv, 4))

weights <- auc_cv / sum(auc_cv)

stack_te <- (prob_rf_te      * weights["RF"]       +
               prob_dart_te    * weights["DART"]     +
               prob_gblinear_te* weights["GBLINEAR"] +
               prob_svm_te     * weights["SVM"]      +
               prob_gbm_te     * weights["GBM"]      +
               prob_glmnet_te  * weights["GLMNET"]   +
               prob_lr_no_te   * weights["LR"])

stack_tr <- (prob_rf_tr      * weights["RF"]       +
               prob_dart_tr    * weights["DART"]     +
               prob_gblinear_tr* weights["GBLINEAR"] +
               prob_svm_tr     * weights["SVM"]      +
               prob_gbm_tr     * weights["GBM"]      +
               prob_glmnet_tr  * weights["GLMNET"]   +
               prob_lr_no_tr   * weights["LR"])

cat("AUC Ensemble Stack (Teste):", round(avaliar_prob(stack_te, obs_te)$auc, 4), "\n")

############################################################
# 20. OTIMIZAÇÃO DE THRESHOLD (Índice de Youden)
############################################################
# O threshold padrão de 0.5 foi definido arbitrariamente e não é ideal
# para dados desbalanceados. O Índice de Youden (J = Sensibilidade +
# Especificidade − 1) maximiza o tradeoff entre os dois, encontrando
# o ponto da curva ROC de maior distância da linha aleatória.
#
# IMPORTANTE: os thresholds são calibrados no TREINO e aplicados no TESTE.
# Calibrar no teste seria data leakage e inflaria artificialmente os resultados.

cat("\n[Threshold] Calibrando thresholds no treino (Youden)...\n")

thr_rf       <- optimize_threshold(obs_tr, prob_rf_tr)
thr_dart     <- optimize_threshold(obs_tr, prob_dart_tr)
thr_gblinear <- optimize_threshold(obs_tr, prob_gblinear_tr)
thr_svm      <- optimize_threshold(obs_tr, prob_svm_tr)
thr_gbm      <- optimize_threshold(obs_tr, prob_gbm_tr)
thr_glmnet   <- optimize_threshold(obs_tr, prob_glmnet_tr)
thr_lr_no    <- optimize_threshold(obs_tr, prob_lr_no_tr)
thr_stack    <- optimize_threshold(obs_tr, stack_tr)

cat("Thresholds (Youden):\n")
cat(sprintf("RF=%.3f | DART=%.3f | GBLinear=%.3f | SVM=%.3f\nGBM=%.3f | ElasticNet=%.3f | Regressão Logística=%.3f | Stack=%.3f\n",
            thr_rf, thr_dart, thr_gblinear, thr_svm,
            thr_gbm, thr_glmnet, thr_lr_no, thr_stack))

############################################################
# 21. TABELAS COMPARATIVAS — TREINO vs TESTE
############################################################
# [FIX 2] Relatamos métricas nos dois conjuntos (treino e teste) para
# cada modelo. Isso permite:
#   • Diagnosticar overfitting (gap treino-teste grande)
#   • Comparar com os colegas (que reportam apenas acurácia de CV)

cat("\n====== TABELA COMPARATIVA: TREINO vs TESTE (threshold = 0.50) ======\n")

tab_split_base <- bind_rows(
  extrair_metricas_split(prob_rf_tr,      obs_tr, prob_rf_te,      obs_te, "Random Forest"),
  extrair_metricas_split(prob_dart_tr,    obs_tr, prob_dart_te,    obs_te, "XGB DART"),
  extrair_metricas_split(prob_gblinear_tr,obs_tr, prob_gblinear_te,obs_te, "XGB GBLinear"),
  extrair_metricas_split(prob_svm_tr,     obs_tr, prob_svm_te,     obs_te, "SVM Radial"),
  extrair_metricas_split(prob_gbm_tr,     obs_tr, prob_gbm_te,     obs_te, "GBM"),
  extrair_metricas_split(prob_glmnet_tr,  obs_tr, prob_glmnet_te,  obs_te, "Elastic Net"),
  extrair_metricas_split(prob_lr_no_tr,   obs_tr, prob_lr_no_te,   obs_te, "Regressão Logística"),
  extrair_metricas_split(stack_tr,        obs_tr, stack_te,        obs_te, "Ensemble Stack")
) %>% arrange(desc(Test_AUC), desc(Test_Acc))

print(tab_split_base)

cat("\n====== TABELA COMPARATIVA: TREINO vs TESTE (threshold otimizado - Youden) ======\n")

tab_split_thr <- bind_rows(
  extrair_metricas_split(prob_rf_tr,      obs_tr, prob_rf_te,      obs_te, "Random Forest",   thr = thr_rf),
  extrair_metricas_split(prob_dart_tr,    obs_tr, prob_dart_te,    obs_te, "XGB DART",         thr = thr_dart),
  extrair_metricas_split(prob_gblinear_tr,obs_tr, prob_gblinear_te,obs_te, "XGB GBLinear",     thr = thr_gblinear),
  extrair_metricas_split(prob_svm_tr,     obs_tr, prob_svm_te,     obs_te, "SVM Radial",       thr = thr_svm),
  extrair_metricas_split(prob_gbm_tr,     obs_tr, prob_gbm_te,     obs_te, "GBM",              thr = thr_gbm),
  extrair_metricas_split(prob_glmnet_tr,  obs_tr, prob_glmnet_te,  obs_te, "Elastic Net",      thr = thr_glmnet),
  extrair_metricas_split(prob_lr_no_tr,   obs_tr, prob_lr_no_te,   obs_te, "Regressão Logística", thr = thr_lr_no),
  extrair_metricas_split(stack_tr,        obs_tr, stack_te,        obs_te, "Ensemble Stack",   thr = thr_stack)
) %>% arrange(desc(Test_AUC), desc(Test_Acc))

print(tab_split_thr)

############################################################
# 22. TABELA CONSOLIDADA (apenas TESTE) — compatível com versão anterior
############################################################

extrair_metricas_teste <- function(prob_te, obs_te, nome, thr = 0.5) {
  res <- avaliar_prob(prob_te, obs_te, thr)
  data.frame(
    Modelo      = nome,
    Threshold   = round(thr, 3),
    Accuracy    = round(as.numeric(res$cm$overall["Accuracy"]),       3),
    AUC         = round(res$auc,                                       3),
    Sensitivity = round(as.numeric(res$cm$byClass["Sensitivity"]),    3),
    Specificity = round(as.numeric(res$cm$byClass["Specificity"]),    3),
    Precision   = round(as.numeric(res$cm$byClass["Pos Pred Value"]), 3),
    F1          = round(f1_score(res$cm),                             3),
    stringsAsFactors = FALSE
  )
}

tab_base <- bind_rows(
  extrair_metricas_teste(prob_rf_te,      obs_te, "Random Forest"),
  extrair_metricas_teste(prob_dart_te,    obs_te, "XGB DART"),
  extrair_metricas_teste(prob_gblinear_te,obs_te, "XGB GBLinear"),
  extrair_metricas_teste(prob_svm_te,     obs_te, "SVM Radial"),
  extrair_metricas_teste(prob_gbm_te,     obs_te, "GBM"),
  extrair_metricas_teste(prob_glmnet_te,  obs_te, "Elastic Net (LR)"),
  extrair_metricas_teste(prob_lr_no_te,   obs_te, "Regressão Logística"),
  extrair_metricas_teste(stack_te,        obs_te, "Ensemble Stack")
) %>% arrange(desc(AUC), desc(Accuracy))

tab_thr <- bind_rows(
  extrair_metricas_teste(prob_rf_te,      obs_te, "Random Forest (thr opt.)",    thr_rf),
  extrair_metricas_teste(prob_dart_te,    obs_te, "XGB DART (thr opt.)",         thr_dart),
  extrair_metricas_teste(prob_gblinear_te,obs_te, "XGB GBLinear (thr opt.)",     thr_gblinear),
  extrair_metricas_teste(prob_svm_te,     obs_te, "SVM Radial (thr opt.)",       thr_svm),
  extrair_metricas_teste(prob_gbm_te,     obs_te, "GBM (thr opt.)",              thr_gbm),
  extrair_metricas_teste(prob_glmnet_te,  obs_te, "Elastic Net (thr opt.)",      thr_glmnet),
  extrair_metricas_teste(prob_lr_no_te,   obs_te, "Regressão Logística (thr opt.)",     thr_lr_no),
  extrair_metricas_teste(stack_te,        obs_te, "Ensemble Stack (thr opt.)",   thr_stack)
) %>% arrange(desc(AUC), desc(Accuracy))

############################################################
# 23. IDENTIFICAR MELHOR MODELO
############################################################

todos <- bind_rows(tab_base, tab_thr) %>%
  arrange(desc(AUC), desc(Accuracy), desc(F1))

best_name <- todos$Modelo[1]
cat("\n>>> MELHOR MODELO:", best_name, "<<<\n")
cat("AUC:", todos$AUC[1], "| Acc:", todos$Accuracy[1],
    "| Sensitivity:", todos$Sensitivity[1],
    "| Specificity:", todos$Specificity[1],
    "| F1:", todos$F1[1], "\n")

best_prob_te <- switch(
  gsub(" \\(thr opt\\.\\)", "", best_name),
  "Random Forest"    = prob_rf_te,
  "XGB DART"         = prob_dart_te,
  "XGB GBLinear"     = prob_gblinear_te,
  "SVM Radial"       = prob_svm_te,
  "GBM"              = prob_gbm_te,
  "Elastic Net"      = prob_glmnet_te,
  "Regressão Logística" = prob_lr_no_te,
  "Ensemble Stack"   = stack_te,
  stack_te
)
best_thr  <- todos$Threshold[1]
best_pred <- factor(ifelse(best_prob_te > best_thr, "Yes", "No"), levels = c("No", "Yes"))
cm_best   <- confusionMatrix(best_pred, obs_te, positive = "Yes")
roc_best  <- roc(obs_te, best_prob_te, quiet = TRUE)

############################################################
# 24. GRÁFICOS
############################################################

# ── 24.1 Comparativo Base (Teste) ──
metricas_long <- tab_base %>%
  dplyr::select(Modelo, Accuracy, AUC, F1) %>%
  pivot_longer(cols = c(Accuracy, AUC, F1), names_to = "Metrica", values_to = "Valor")

p_base_chart <- ggplot(metricas_long,
                       aes(x = reorder(Modelo, Valor), y = Valor, fill = Metrica)) +
  geom_col(position = position_dodge(0.8), width = 0.7) +
  scale_y_continuous(limits = c(0, 1)) +
  coord_flip() +
  theme_minimal(base_size = 11) +
  labs(title = "Comparação Base dos Algoritmos — Conjunto de TESTE (thr = 0.50)",
       x = NULL, y = "Score")
print(p_base_chart)

# ── 24.2 Treino vs Teste por modelo (gap de overfitting) ──
gap_df <- tab_split_base %>%
  dplyr::select(Modelo, Train_AUC, Test_AUC) %>%
  pivot_longer(cols = c(Train_AUC, Test_AUC), names_to = "Conjunto", values_to = "AUC") %>%
  mutate(Conjunto = ifelse(Conjunto == "Train_AUC", "Treino", "Teste"))

p_gap <- ggplot(gap_df, aes(x = reorder(Modelo, AUC), y = AUC, fill = Conjunto)) +
  geom_col(position = position_dodge(0.7), width = 0.65) +
  scale_fill_manual(values = c("Treino" = "#2E8B57", "Teste" = "#4472C4")) +
  coord_flip() +
  theme_minimal(base_size = 11) +
  labs(
    title    = "AUC Treino vs Teste por Modelo",
    subtitle = "Gap pequeno = boa generalização | Gap grande = overfitting",
    x = NULL, y = "AUC", fill = "Conjunto"
  )
print(p_gap)

# ── 24.3 Comparativo threshold otimizado ──
metricas_thr_long <- tab_thr %>%
  mutate(Modelo = gsub(" \\(thr opt\\.\\)", "", Modelo)) %>%
  dplyr::select(Modelo, Accuracy, AUC, F1) %>%
  pivot_longer(cols = c(Accuracy, AUC, F1), names_to = "Metrica", values_to = "Valor")

p_thr_chart <- ggplot(metricas_thr_long,
                      aes(x = reorder(Modelo, Valor), y = Valor, fill = Metrica)) +
  geom_col(position = position_dodge(0.8), width = 0.7) +
  scale_y_continuous(limits = c(0, 1)) +
  coord_flip() +
  theme_minimal(base_size = 11) +
  labs(title = "Comparação com Threshold Otimizado (Youden) — Conjunto de TESTE",
       x = NULL, y = "Score")
print(p_thr_chart)

# ── 24.4 Curva ROC do Melhor Modelo ──
roc_df <- data.frame(FPR = 1 - roc_best$specificities, TPR = roc_best$sensitivities)

p_roc <- ggplot(roc_df, aes(x = FPR, y = TPR)) +
  geom_line(color = "firebrick", linewidth = 1.2) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray60") +
  theme_minimal(base_size = 12) +
  labs(title    = paste0("Curva ROC — ", best_name),
       subtitle = paste0("AUC = ", round(as.numeric(auc(roc_best)), 4)),
       x = "1 - Especificidade (FPR)", y = "Sensibilidade (TPR)")
print(p_roc)

# ── 24.5 Matriz de Confusão ──
cm_df <- as.data.frame(cm_best$table)
colnames(cm_df) <- c("Predito", "Real", "Freq")

p_cm <- ggplot(cm_df, aes(x = Predito, y = Real, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), color = "white", size = 6, fontface = "bold") +
  scale_fill_gradient(low = "#5DA5DA", high = "#0B3C5D") +
  theme_minimal(base_size = 12) +
  labs(title    = paste0("Matriz de Confusão — ", best_name),
       subtitle = paste0("Threshold = ", round(best_thr, 3)))
print(p_cm)

# ── 24.6 Importância das Variáveis (RF e XGB DART) ──
imp_df <- varImp(model_rf, scale = TRUE)$importance %>%
  rownames_to_column("Variavel") %>%
  mutate(Importancia = rowMeans(dplyr::select(., -Variavel))) %>%
  dplyr::select(Variavel, Importancia) %>%
  arrange(desc(Importancia)) %>% slice_head(n = 15)

p_imp <- ggplot(imp_df, aes(x = reorder(Variavel, Importancia), y = Importancia)) +
  geom_col(fill = "#2E8B57", width = 0.8) +
  coord_flip() + theme_minimal(base_size = 11) +
  labs(title = "Importância das Variáveis — Random Forest", x = NULL, y = "Importância")
print(p_imp)

imp_dart <- xgb.importance(model = model_dart) %>% as.data.frame() %>% slice_head(n = 15)

p_imp_dart <- ggplot(imp_dart, aes(x = reorder(Feature, Gain), y = Gain)) +
  geom_col(fill = "#E07B39", width = 0.8) +
  coord_flip() + theme_minimal(base_size = 11) +
  labs(title = "Importância das Variáveis — XGB DART", x = NULL, y = "Gain")
print(p_imp_dart)

# ── 24.7 Impacto do Limiar no Melhor Modelo ──
thr_grid <- seq(0.05, 0.95, by = 0.01)
thr_impact <- lapply(thr_grid, function(thr) {
  p  <- factor(ifelse(best_prob_te > thr, "Yes", "No"), levels = c("No", "Yes"))
  cm <- confusionMatrix(p, obs_te, positive = "Yes")
  data.frame(Threshold = thr,
             Accuracy    = as.numeric(cm$overall["Accuracy"]),
             Sensitivity = as.numeric(cm$byClass["Sensitivity"]),
             Specificity = as.numeric(cm$byClass["Specificity"]),
             F1          = f1_score(cm))
}) %>% bind_rows() %>%
  pivot_longer(c(Accuracy, Sensitivity, Specificity, F1),
               names_to = "Metrica", values_to = "Valor")

p_thr_impact <- ggplot(thr_impact, aes(x = Threshold, y = Valor, color = Metrica)) +
  geom_line(linewidth = 1) +
  geom_vline(xintercept = best_thr, linetype = "dashed", color = "black") +
  theme_minimal(base_size = 11) +
  labs(title    = paste0("Impacto do Limiar — ", best_name),
       subtitle = paste0("Limiar otimizado = ", round(best_thr, 3)),
       x = "Threshold", y = "Score", color = "Métrica")
print(p_thr_impact)

############################################################
# 25. COMPARAÇÃO COM O ARTIGO DOS COLEGAS
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
cat(sprintf("Nosso melhor Accuracy (base):         %.1f%%\n", max(tab_base$Accuracy) * 100))
cat(sprintf("Nosso melhor Accuracy (thr opt.):     %.1f%%\n", max(tab_thr$Accuracy) * 100))
cat(sprintf("Nosso melhor AUC:                     %.4f\n",   max(tab_thr$AUC)))
cat(sprintf("Nosso melhor F1:                      %.3f\n",   max(tab_thr$F1)))
cat(sprintf("Regressão Logística — Accuracy (base):       %.1f%%\n",
            tab_base$Accuracy[tab_base$Modelo == "Regressão Logística"] * 100))
cat(sprintf("Regressão Logística — Accuracy (thr opt.):   %.1f%%\n",
            tab_thr$Accuracy[grepl("Regressão Logística", tab_thr$Modelo)] * 100))

############################################################
# 26. RESULTADO FINAL CONSOLIDADO
############################################################

cat("\n")
cat("=============================================================\n")
cat("        RESULTADOS FINAIS — ILPD (seed=123, 80/20)          \n")
cat("=============================================================\n")

cat("\n--- Tabela 1: Treino vs Teste — baseline (thr = 0.50) ---\n")
print(tab_split_base)

cat("\n--- Tabela 2: Treino vs Teste — threshold otimizado (Youden) ---\n")
print(tab_split_thr)

cat("\n--- Tabela 3: Apenas Teste — baseline ---\n")
print(tab_base)

cat("\n--- Tabela 4: Apenas Teste — threshold otimizado ---\n")
print(tab_thr)

cat("\n--- Melhor Modelo ---\n")
cat("Modelo:", best_name, "\n")
cat("AUC:", todos$AUC[1],
    "| Accuracy:", todos$Accuracy[1],
    "| Sensitivity:", todos$Sensitivity[1],
    "| Specificity:", todos$Specificity[1],
    "| F1:", todos$F1[1], "\n")
cat("=============================================================\n")