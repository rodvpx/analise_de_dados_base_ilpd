# ============================================================
# Análise: Indian Liver Patient Dataset (ILPD)
# Estratégia: seleção padronizada de variáveis, avaliação treino/teste,
#             feature engineering e ensemble com stacking
# ============================================================

############################################################
# 0. INSTALAÇÃO DE PACOTES (EXECUTAR UMA VEZ SE NECESSÁRIO)
############################################################

# Função auxiliar para instalar e carregar pacotes automaticamente
# install_if_missing <- function(p) {
#   if (!require(p, character.only = TRUE)) install.packages(p)
# }
# lapply(c("dplyr", "ggplot2", "caret", "randomForest", "e1071",
#          "xgboost", "pROC", "tidyr", "MASS", "glmnet",
#          "tibble", "gridExtra"), install_if_missing)

############################################################
# 1. PACOTES
############################################################

# Carregamento das bibliotecas essenciais para manipulação de dados,
# modelagem preditiva, cálculos de métricas e visualização.
library(dplyr)        # Manipulação de dados
library(ggplot2)      # Visualização gráfica
library(caret)        # Pipeline de Machine Learning
library(randomForest) # Algoritmo Random Forest
library(e1071)        # Algoritmo SVM (Support Vector Machine)
library(xgboost)      # Algoritmo XGBoost (Gradient Boosting)
library(pROC)         # Análise de curvas ROC e AUC
library(tidyr)        # Manipulação e pivotamento de tabelas
library(MASS)         # Funções estatísticas (ex: StepAIC)
library(glmnet)       # Algoritmo Elastic Net
library(tibble)       # Estruturas de dados (DataFrames modernos)
library(gridExtra)    # Arranjos gráficos complexos

############################################################
# 2. CARREGAMENTO DOS DADOS
############################################################

path_base <- "Indian Liver Patient Dataset (ILPD).csv"

# Definindo nomes descritivos para as colunas com base na documentação do dataset
col_names <- c("Age", "Gender", "TB", "DB", "Alkphos",
               "SGPT", "SGOT", "TP", "ALB", "AG_Ratio", "Selector")

# Carrega o arquivo CSV sem cabeçalho original, atribuindo os nomes definidos
df <- read.csv(path_base, header = FALSE, col.names = col_names,
               stringsAsFactors = FALSE)

cat("Dimensões originais:", nrow(df), "x", ncol(df), "\n")
cat("Distribuição da variável alvo:\n")
print(table(df$Selector))

############################################################
# 3. PRÉ-PROCESSAMENTO
############################################################

# Removemos linhas idênticas para evitar viés no treinamento
df <- df[!duplicated(df), ]
cat("Após remoção de duplicatas:", nrow(df), "linhas\n")

# Transformamos a variável 'Gender' categórica para formato binário numérico
df$Gender <- ifelse(df$Gender == "Male", 1L, 0L)

# Identificamos e imputamos valores ausentes em AG_Ratio
# Usamos a mediana pois ela é mais robusta a outliers do que a média.
if (any(is.na(df$AG_Ratio))) {
  med_ag <- median(df$AG_Ratio, na.rm = TRUE)
  df$AG_Ratio[is.na(df$AG_Ratio)] <- med_ag
}

# A variável alvo é convertida de 1/2 para "Yes" (Doente) e "No" (Saudável)
# Garantimos que seja tratada como fator para problemas de classificação.
df$Selector <- ifelse(df$Selector == 1, "Yes", "No")
df$Selector <- factor(df$Selector, levels = c("No", "Yes"))

target <- "Selector"

############################################################
# 4. FEATURE ENGINEERING MÉDICA
############################################################
# Criamos novas variáveis baseadas em conhecimento clínico hepatológico 
# para enriquecer os dados e fornecer melhores sinais aos modelos.

eps <- 1e-6 # Valor ínfimo para evitar divisão por zero

# Razões clássicas entre enzimas
df$AST_ALT_ratio   <- df$SGOT / (df$SGPT + eps)
df$Bilirubin_ratio <- df$DB / (df$TB + eps)
df$Protein_ratio   <- df$ALB / (df$TP + eps)

df <- df %>%
  mutate(
    # Razão TB/DB para checar hiperbilirrubinemia
    TB_DB_ratio     = ifelse(DB > 0, TB / DB, TB),
    # Relação SGPT/SGOT ajuda a detectar danos específicos hepáticos
    SGPT_SGOT_ratio = ifelse(SGOT > 0, SGPT / SGOT, SGPT),
    # Bilirrubina Indireta (Total - Direta)
    Indirect_Bili   = TB - DB,
    # Globulina: diferença entre Proteínas Totais e Albumina
    Globulin        = TP - ALB,
    
    # Aplicação de log(1+x) para normalizar a distribuição 
    # extremamente assimétrica das enzimas hepáticas
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

# Garantimos reprodutibilidade através da semente
set.seed(123)

# 'createDataPartition' faz um particionamento estratificado, 
# preservando a mesma proporção da classe alvo no treino e teste.
train_index <- createDataPartition(df[[target]], p = 0.80, list = FALSE)

train_raw <- df[train_index, ]
test_raw  <- df[-train_index, ]

cat("\nTreino:", nrow(train_raw), "| Teste:", nrow(test_raw), "\n")

############################################################
# 6. SELEÇÃO DE VARIÁVEIS — StepAIC backward
############################################################

predictor_names <- setdiff(names(train_raw), target)

# Remove variáveis com "Near Zero Variance" (que quase não mudam de valor)
# pois não agregam poder discriminativo.
nzv_idx <- nearZeroVar(train_raw[, predictor_names])
if (length(nzv_idx) > 0) {
  cat("Removidas por NZV:", length(nzv_idx), "variáveis\n")
  predictor_names <- predictor_names[-nzv_idx]
}

# Remove uma dentre variáveis altamente correlacionadas (>95%) 
# para evitar redundância de informações
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

cat("\n[Seleção de Variáveis] Executando StepAIC backward no treino original...\n")

# Função para realizar seleção backward usando o Critério de Informação de Akaike (AIC)
get_backward_vars <- function(train_df, target_name) {
  # Ajusta o modelo completo com todas as variáveis
  full_model  <- glm(as.formula(paste(target_name, "~ .")),
                     data = train_df, family = binomial())
  # stepAIC remove variáveis iterativamente até chegar no melhor conjunto
  step_model  <- stepAIC(full_model, direction = "backward", trace = FALSE)
  vars <- setdiff(names(coef(step_model)), "(Intercept)")
  unique(vars)
}

# Trata erros eventuais na execução do StepAIC
selected_vars <- tryCatch(
  get_backward_vars(train_raw[, c(target, predictor_names)], target),
  error = function(e) {
    cat("StepAIC falhou, usando todas as variáveis:", conditionMessage(e), "\n")
    predictor_names
  }
)
selected_vars <- intersect(selected_vars, predictor_names)

# Reverte fallback se remover demais (menos de 3 vars restantes)
if (length(selected_vars) < 3) selected_vars <- predictor_names

cat("Variáveis selecionadas (StepAIC):", length(selected_vars), "\n")
cat(paste(selected_vars, collapse = ", "), "\n")

# Criamos a base final de modelagem, com somente as selecionadas
train_sel <- train_raw[, c(target, selected_vars)]
test_sel  <- test_raw[,  c(target, selected_vars)]

############################################################
# 7. NORMALIZAÇÃO (APENAS COM TREINO!)
############################################################

# Center e scale: subtrai a média e divide pelo desvio padrão.
# O cálculo de pré-processamento DEVE basear-se APENAS no treino
# para evitar vazamento de dados (data leakage) do teste.
predictors <- setdiff(names(train_sel), target)
preproc <- preProcess(train_sel[, predictors],
                      method = c("center","scale"))

train_norm <- predict(preproc, train_sel)
test_norm  <- predict(preproc, test_sel)

############################################################
# 8. TRATAMENTO DO DESBALANCEAMENTO
############################################################

# Em vez de balanceamento por over/undersampling (como SMOTE), 
# usaremos 'Class Weights' para não modificar os dados reais
class_tab <- table(train_sel[[target]])
class_weights <- as.numeric(sum(class_tab) / (length(class_tab) * class_tab))
names(class_weights) <- names(class_tab)

# Calcula o peso da observação: a classe minoritária recebe peso maior
obs_weights_train <- ifelse(train_sel[[target]] == "No",
                            class_weights["No"],
                            class_weights["Yes"])

############################################################
# 9. CONTROLE DE TREINAMENTO (Repeated CV)
############################################################

# Configuração global de validação cruzada para o caret.
# 10 Folds repetidos 5 vezes gera estimativas mais precisas/estáveis.
# Foco explícito em métricas binárias (ROC/AUC).
train_ctrl <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

############################################################
# 10. FUNÇÕES AUXILIARES
############################################################

# Função para cálculo do F1-Score a partir da Matriz de Confusão
f1_score <- function(cm) {
  p <- as.numeric(cm$byClass["Pos Pred Value"])
  r <- as.numeric(cm$byClass["Sensitivity"])
  if (is.na(p) || is.na(r) || (p + r) == 0) return(0)
  2 * (p * r) / (p + r)
}

# Encontra o limite (threshold) de corte ótimo (maximizando J de Youden)
optimize_threshold <- function(obs, prob) {
  roc_obj <- roc(obs, prob, quiet = TRUE)
  thr_df  <- coords(roc_obj, x = "best", best.method = "youden",
                    ret = "threshold", transpose = FALSE)
  thr <- as.numeric(thr_df[1, "threshold"])
  if (is.na(thr) || length(thr) == 0) thr <- 0.5
  thr
}

# Avalia as probabilidades dada uma linha de corte (thr)
avaliar_prob <- function(prob, obs, thr = 0.5) {
  pred    <- factor(ifelse(prob > thr, "Yes", "No"), levels = c("No", "Yes"))
  cm      <- confusionMatrix(pred, obs, positive = "Yes")
  roc_obj <- roc(obs, prob, quiet = TRUE)
  list(cm = cm, auc = as.numeric(auc(roc_obj)), threshold = thr, prob = prob)
}

# Extrai e formata métricas gerando os dados tabulares (tanto p/ treino como teste)
extrair_metricas_split <- function(prob_tr, obs_tr, prob_te, obs_te, nome, thr = 0.5) {
  res_tr <- avaliar_prob(prob_tr, obs_tr, thr)
  res_te <- avaliar_prob(prob_te, obs_te, thr)
  data.frame(
    Modelo       = nome,
    Threshold    = round(thr, 3),
    Train_Acc    = round(as.numeric(res_tr$cm$overall["Accuracy"]),       3),
    Train_AUC    = round(res_tr$auc,                                       3),
    Train_Sens   = round(as.numeric(res_tr$cm$byClass["Sensitivity"]),    3),
    Train_Spec   = round(as.numeric(res_tr$cm$byClass["Specificity"]),    3),
    Train_F1     = round(f1_score(res_tr$cm),                             3),
    Test_Acc     = round(as.numeric(res_te$cm$overall["Accuracy"]),       3),
    Test_AUC     = round(res_te$auc,                                       3),
    Test_Sens    = round(as.numeric(res_te$cm$byClass["Sensitivity"]),    3),
    Test_Spec    = round(as.numeric(res_te$cm$byClass["Specificity"]),    3),
    Test_F1      = round(f1_score(res_te$cm),                             3),
    stringsAsFactors = FALSE
  )
}

# Extrai e formata as métricas isoladamente para o conjunto de teste
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

############################################################
# 11. MODELO 1 — RANDOM FOREST (com pesos)
############################################################
cat("\n[1/5] Treinando Random Forest...\n")
set.seed(123)

# RandomForest comumente dispensa normalização. Usamos train_sel (não-norm).
# 'classwt' lida com as classes desbalanceadas.
model_rf <- train(
  Selector ~ .,
  data = train_sel,
  method = "rf",
  metric = "ROC",
  trControl = train_ctrl,
  ntree = 800,
  classwt = c("No" = 2.5, "Yes" = 1)
)

############################################################
# 12. MODELO 2 — ELASTIC NET
############################################################
cat("\n[2/5] Treinando Elastic Net...\n")
set.seed(123)

# Regressão linear regularizada via glmnet, sensível a escala (usando train_norm)
# Testamos combinações de L1 e L2 na grid search (alpha = 0 a 1).
grid_glmnet <- expand.grid(
  alpha  = seq(0,1,0.1),
  lambda = 10^seq(-4,1,length=20)
)

model_enet <- train(
  Selector ~ .,
  data = train_norm,
  method = "glmnet",
  metric = "ROC",
  trControl = train_ctrl,
  tuneGrid = grid_glmnet
)

############################################################
# 13. MODELO 3 — REGRESSÃO LOGÍSTICA
############################################################
cat("\n[3/5] Treinando Regressão Logística...\n")
set.seed(123)

# Regressão logística tradicional como baseline interpretável (usando train_norm)
model_log <- train(
  Selector ~ .,
  data = train_norm,
  method = "glm",
  family = "binomial",
  metric = "ROC",
  trControl = train_ctrl
)

############################################################
# 14. MODELO 4 — SVM
############################################################
cat("\n[4/5] Treinando SVM...\n")
set.seed(123)

# SVM com kernel radial, também sensível a escala.
# Testamos sigmas (suavidade da fronteira) e Custos (tolerância a margem)
grid_svm <- expand.grid(
  sigma = c(0.01,0.05,0.1,0.2),
  C     = c(0.25,0.5,1,2,4)
)

model_svm <- train(
  Selector ~ .,
  data = train_norm,
  method = "svmRadial",
  metric = "ROC",
  trControl = train_ctrl,
  tuneGrid = grid_svm
)

############################################################
# 15. MODELO 5 — XGBOOST (Grid Search + Platt Scaling)
############################################################
cat("\n[5/5] Treinando XGBoost (gbtree)...\n")
set.seed(123)

# Converte os dados para matrizes esparsas otimizadas próprias do XGBoost
x_train <- model.matrix(Selector ~ . -1, train_sel)
x_test  <- model.matrix(Selector ~ . -1, test_sel)

y_train <- ifelse(train_sel$Selector == "Yes", 1, 0)
y_test  <- ifelse(test_sel$Selector == "Yes", 1, 0)

dtrain <- xgb.DMatrix(data = x_train, label = y_train)
dtest  <- xgb.DMatrix(data = x_test,  label = y_test)

# Definimos um Grid massivo para afinar hiperparâmetros
grid <- expand.grid(
  eta = c(0.01, 0.05, 0.1),
  max_depth = c(2,3,4),
  min_child_weight = c(1,3,5),
  subsample = c(0.7, 0.85, 1),
  colsample_bytree = c(0.7, 0.85, 1)
)

best_auc <- 0
best_params <- list()
best_nrounds <- 300   # valor default seguro

# Looping explorando e treinando validações cruzadas
for(i in 1:nrow(grid)){
  
  params <- list(
    booster = "gbtree",
    objective = "binary:logistic",
    eval_metric = "auc",
    eta = grid$eta[i],
    max_depth = grid$max_depth[i],
    min_child_weight = grid$min_child_weight[i],
    subsample = grid$subsample[i],
    colsample_bytree = grid$colsample_bytree[i]
  )
  
  # xgb.cv descobre o número exato de iterações sem overfit (early_stopping)
  cv <- xgb.cv(
    params = params,
    data = dtrain,
    nrounds = 1000,
    nfold = 10,
    stratified = TRUE,
    early_stopping_rounds = 50,
    maximize = TRUE,
    verbose = 0
  )
  
  auc <- max(cv$evaluation_log$test_auc_mean)

  nrounds_cv <- ifelse(is.null(cv$best_iteration),
                       which.max(cv$evaluation_log$test_auc_mean),
                       cv$best_iteration)
  
  # Salva o melhor parâmetro encontrado até agora
  if(auc > best_auc){
    best_auc <- auc
    best_params <- params
    best_nrounds <- nrounds_cv
  }
}

cat("Melhor AUC CV XGB:", best_auc, "\n")
cat("Melhor nrounds:", best_nrounds, "\n")

# Treina o modelo final do XGBoost usando os hyperparametros descobertos
model_xgb <- xgb.train(
  params = best_params,
  data = dtrain,
  nrounds = best_nrounds,
  verbose = 0
)

# ============================================
# Calibração de probabilidade (Platt Scaling)
# ============================================

# XGBoost entrega scores bons para ranking (AUC), mas probabilidades frequentemente descalibradas.
# Extraímos as previsões puras
prob_train_raw <- predict(model_xgb, dtrain)
prob_test_raw  <- predict(model_xgb, dtest)

# Ajustamos uma regressão logística (GLM) transformando os raw scores em "Probabilidades Reais"
calibration_model <- glm(
  y_train ~ prob_train_raw,
  family = binomial
)

# Aplicar o modelo de calibração para gerar predições escalonadas corretas
prob_xgb_train <- predict(
  calibration_model,
  newdata = data.frame(prob_train_raw = prob_train_raw),
  type = "response"
)

prob_xgb_test <- predict(
  calibration_model,
  newdata = data.frame(prob_train_raw = prob_test_raw),
  type = "response"
)

############################################################
# 16. PREDIÇÕES (TREINO E TESTE)
############################################################

obs_te <- test_sel[[target]]
obs_tr <- train_sel[[target]]

# Coletando a probabilidade ('Yes') prevista pelos demais modelos
prob_rf_te   <- predict(model_rf,   test_sel,  type = "prob")[, "Yes"]
prob_enet_te <- predict(model_enet, test_norm, type = "prob")[, "Yes"]
prob_log_te  <- predict(model_log,  test_norm, type = "prob")[, "Yes"]
prob_svm_te  <- predict(model_svm,  test_norm, type = "prob")[, "Yes"]

prob_rf_tr   <- predict(model_rf,   train_sel, type = "prob")[, "Yes"]
prob_enet_tr <- predict(model_enet, train_norm, type = "prob")[, "Yes"]
prob_log_tr  <- predict(model_log,  train_norm, type = "prob")[, "Yes"]
prob_svm_tr  <- predict(model_svm,  train_norm, type = "prob")[, "Yes"]

############################################################
# 17. ENSEMBLE STACKING (ponderado por AUC)
############################################################

# Reune o máximo do Cross Validation de cada modelo para utilizarmos como peso
auc_cv <- c(
  RF       = max(model_rf$results$ROC),
  ENET     = max(model_enet$results$ROC),
  LOG      = max(model_log$results$ROC),
  SVM      = max(model_svm$results$ROC),
  XGB      = best_auc
)

cat("\nAUC de CV/Teste por modelo (pesos do ensemble):\n")
print(round(auc_cv, 4))

# Converte os AUCs em proporções
weights <- auc_cv / sum(auc_cv)

# Modelo final combinando os 5 preditores dando mais peso aos de melhor AUC
stack_te <- (prob_rf_te    * weights["RF"] +
             prob_enet_te  * weights["ENET"] +
             prob_log_te   * weights["LOG"] +
             prob_svm_te   * weights["SVM"] +
             prob_xgb_test * weights["XGB"])

stack_tr <- (prob_rf_tr     * weights["RF"] +
             prob_enet_tr   * weights["ENET"] +
             prob_log_tr    * weights["LOG"] +
             prob_svm_tr    * weights["SVM"] +
             prob_xgb_train * weights["XGB"])

cat("AUC Ensemble Stack (Teste):", round(avaliar_prob(stack_te, obs_te)$auc, 4), "\n")

############################################################
# 18. OTIMIZAÇÃO DE THRESHOLD (Índice de Youden)
############################################################
# Modelos predizem de 0 a 1. Na medicina 50% pode ser arriscado.
# O 'Índice de Youden' acha o limiar que equilibra Falso Positivo e Falso Negativo.
cat("\n[Threshold] Calibrando thresholds no treino (Youden)...\n")

thr_rf   <- optimize_threshold(obs_tr, prob_rf_tr)
thr_enet <- optimize_threshold(obs_tr, prob_enet_tr)
thr_log  <- optimize_threshold(obs_tr, prob_log_tr)
thr_svm  <- optimize_threshold(obs_tr, prob_svm_tr)
thr_xgb  <- optimize_threshold(obs_tr, prob_xgb_train)
thr_stack<- optimize_threshold(obs_tr, stack_tr)

cat("Thresholds (Youden):\n")
cat(sprintf("RF=%.3f | ElasticNet=%.3f | Reg.Log=%.3f | SVM=%.3f | XGB=%.3f | Stack=%.3f\n",
            thr_rf, thr_enet, thr_log, thr_svm, thr_xgb, thr_stack))

############################################################
# GRÁFICO 1 — Curvas ROC comparando modelos
############################################################
# Avalia a capacidade de distinção entre classes para cada modelo através da Curva ROC
roc_rf    <- roc(obs_te, prob_rf_te, quiet = TRUE)
roc_enet  <- roc(obs_te, prob_enet_te, quiet = TRUE)
roc_log   <- roc(obs_te, prob_log_te, quiet = TRUE)
roc_svm   <- roc(obs_te, prob_svm_te, quiet = TRUE)
roc_xgb   <- roc(obs_te, prob_xgb_test, quiet = TRUE)
roc_stack <- roc(obs_te, stack_te, quiet = TRUE)

plot(roc_rf, col = "green", main = "Curvas ROC - Comparação de Modelos")
lines(roc_enet, col = "blue")
lines(roc_log, col = "cyan")
lines(roc_svm, col = "purple")
lines(roc_xgb, col = "red", lwd = 2)
lines(roc_stack, col = "orange", lwd = 2)
legend("bottomright", legend = c("Random Forest", "Elastic Net", "Reg. Logística", "SVM", "XGBoost", "Ensemble Stack"),
       col = c("green", "blue", "cyan", "purple", "red", "orange"), lwd = 2)

############################################################
# GRÁFICO 2 — Importância das Variáveis (XGBoost)
############################################################
# Analisa quais variáveis o XGBoost utilizou com mais peso nos ganhos de decisão das árvores
importance_matrix <- xgb.importance(feature_names = colnames(dtrain), model = model_xgb)
xgb.plot.importance(importance_matrix, top_n = 10, measure = "Gain", 
                    main = "Importância das Variáveis (XGBoost)")

############################################################
# GRÁFICO 3 — Distribuição das probabilidades previstas
############################################################
# Exibe a separação das probabilidades previstas pelo modelo entre pacientes Doentes x Saudáveis
prob_df <- data.frame(
  Probabilidade = prob_xgb_test,
  ClasseReal = obs_te
)

p_prob <- ggplot(prob_df, aes(x = Probabilidade, fill = ClasseReal)) +
  geom_density(alpha = 0.5) +
  theme_minimal() +
  labs(title = "Distribuição das Probabilidades Previstas (XGBoost)",
       x = "Probabilidade de Doença Hepática",
       y = "Densidade") +
  geom_vline(xintercept = thr_xgb, linetype = "dashed", color = "red")
print(p_prob)

############################################################
# GRÁFICO 4 — Matriz de confusão XGBoost
############################################################
# Gera visualização para os Casos de Falso Positivo/Negativo com base no limiar calibrado
pred_xgb_class <- factor(ifelse(prob_xgb_test > thr_xgb, "Yes", "No"), levels = c("No", "Yes"))
cm_xgb <- confusionMatrix(pred_xgb_class, obs_te, positive = "Yes")

cm_df_xgb <- as.data.frame(cm_xgb$table)
colnames(cm_df_xgb) <- c("Predito", "Real", "Freq")

p_cm <- ggplot(cm_df_xgb, aes(x = Predito, y = Real, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), color = "white", size = 6, fontface = "bold") +
  scale_fill_gradient(low = "#5DA5DA", high = "#0B3C5D") +
  theme_minimal(base_size = 12) +
  labs(title = "Matriz de Confusão — XGBoost",
       subtitle = paste0("Threshold = ", round(thr_xgb, 3)))
print(p_cm)

############################################################
# GRÁFICO 5 — Comparação de desempenho
############################################################
# Exibe um comparativo em barra demonstrando a superioridade (ou igualdade) via AUC entre os modelos
auc_df <- data.frame(
  Modelo = c("Random Forest", "Elastic Net", "Regressão Logística", "SVM Radial", "XGBoost gbtree", "Ensemble Stack"),
  AUC = c(roc_rf$auc, roc_enet$auc, roc_log$auc, roc_svm$auc, roc_xgb$auc, roc_stack$auc)
)

p_auc <- ggplot(auc_df, aes(x = reorder(Modelo, AUC), y = AUC, fill = Modelo)) +
  geom_col(show.legend = FALSE) +
  coord_flip() +
  scale_fill_brewer(palette = "Set2") +
  theme_minimal() +
  labs(title = "Comparação de Desempenho (AUC)",
       x = "Modelo",
       y = "Área sob a Curva (AUC)") +
  geom_text(aes(label = round(AUC, 3)), hjust = -0.2, size = 4) +
  ylim(0, 1)
print(p_auc)

############################################################
# GRÁFICO 6 — Variáveis Selecionadas para Modelagem (StepAIC)
############################################################
# Lista todas as variáveis construídas e destaca quais foram selecionadas
todas_vars <- setdiff(names(train_raw), target)
status_vars <- data.frame(
  Variavel = todas_vars,
  Status = ifelse(todas_vars %in% selected_vars, "Selecionada", "Descartada (StepAIC / NZV / Corr)")
)

p_selecao <- ggplot(status_vars, aes(x = reorder(Variavel, Status == "Selecionada"), y = 1, fill = Status)) +
  geom_tile(color = "white", linewidth = 1) +
  scale_fill_manual(values = c("Selecionada" = "#2E8B57", "Descartada (StepAIC / NZV / Corr)" = "#E07B39")) +
  coord_flip() +
  theme_minimal(base_size = 11) +
  theme(axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        panel.grid = element_blank()) +
  labs(title = "Status das Variáveis Pós-Seleção",
       subtitle = paste("Total Selecionadas:", length(selected_vars)),
       x = "Variável",
       y = NULL,
       fill = "Status")
print(p_selecao)

############################################################
# 19. TABELAS COMPARATIVAS — TREINO vs TESTE
############################################################
# Estas tabelas expõem as métricas de treino contra as de teste.
# Modelos com 90% treino e 70% teste configuram clássicos de 'Overfitting'.
cat("\n====== TABELA COMPARATIVA: TREINO vs TESTE (threshold = 0.50) ======\n")

tab_split_base <- bind_rows(
  extrair_metricas_split(prob_rf_tr,     obs_tr, prob_rf_te,     obs_te, "Random Forest"),
  extrair_metricas_split(prob_enet_tr,   obs_tr, prob_enet_te,   obs_te, "Elastic Net"),
  extrair_metricas_split(prob_log_tr,    obs_tr, prob_log_te,    obs_te, "Regressão Logística"),
  extrair_metricas_split(prob_svm_tr,    obs_tr, prob_svm_te,    obs_te, "SVM Radial"),
  extrair_metricas_split(prob_xgb_train, obs_tr, prob_xgb_test,  obs_te, "XGBoost gbtree"),
  extrair_metricas_split(stack_tr,       obs_tr, stack_te,       obs_te, "Ensemble Stack")
) %>% arrange(desc(Test_AUC), desc(Test_Acc))

print(tab_split_base)

cat("\n====== TABELA COMPARATIVA: TREINO vs TESTE (threshold otimizado - Youden) ======\n")

tab_split_thr <- bind_rows(
  extrair_metricas_split(prob_rf_tr,     obs_tr, prob_rf_te,     obs_te, "Random Forest", thr = thr_rf),
  extrair_metricas_split(prob_enet_tr,   obs_tr, prob_enet_te,   obs_te, "Elastic Net",   thr = thr_enet),
  extrair_metricas_split(prob_log_tr,    obs_tr, prob_log_te,    obs_te, "Regressão Logística", thr = thr_log),
  extrair_metricas_split(prob_svm_tr,    obs_tr, prob_svm_te,    obs_te, "SVM Radial",    thr = thr_svm),
  extrair_metricas_split(prob_xgb_train, obs_tr, prob_xgb_test,  obs_te, "XGBoost gbtree", thr = thr_xgb),
  extrair_metricas_split(stack_tr,       obs_tr, stack_te,       obs_te, "Ensemble Stack",thr = thr_stack)
) %>% arrange(desc(Test_AUC), desc(Test_Acc))

print(tab_split_thr)

############################################################
# 20. TABELA CONSOLIDADA (apenas TESTE)
############################################################
# Resumo focado puramente na predição cega (conjunto teste isolado)

tab_base <- bind_rows(
  extrair_metricas_teste(prob_rf_te,     obs_te, "Random Forest"),
  extrair_metricas_teste(prob_enet_te,   obs_te, "Elastic Net"),
  extrair_metricas_teste(prob_log_te,    obs_te, "Regressão Logística"),
  extrair_metricas_teste(prob_svm_te,    obs_te, "SVM Radial"),
  extrair_metricas_teste(prob_xgb_test,  obs_te, "XGBoost gbtree"),
  extrair_metricas_teste(stack_te,       obs_te, "Ensemble Stack")
) %>% arrange(desc(AUC), desc(Accuracy))

tab_thr <- bind_rows(
  extrair_metricas_teste(prob_rf_te,     obs_te, "Random Forest (thr opt.)",    thr_rf),
  extrair_metricas_teste(prob_enet_te,   obs_te, "Elastic Net (thr opt.)",      thr_enet),
  extrair_metricas_teste(prob_log_te,    obs_te, "Regressão Logística (thr opt.)", thr_log),
  extrair_metricas_teste(prob_svm_te,    obs_te, "SVM Radial (thr opt.)",       thr_svm),
  extrair_metricas_teste(prob_xgb_test,  obs_te, "XGBoost gbtree (thr opt.)", thr_xgb),
  extrair_metricas_teste(stack_te,       obs_te, "Ensemble Stack (thr opt.)",   thr_stack)
) %>% arrange(desc(AUC), desc(Accuracy))

############################################################
# 21. RESULTADO FINAL CONSOLIDADO
############################################################
# Impressão de relatório em tela da melhor combinação geral

cat("\n")
cat("=============================================================\n")
cat("        RESULTADOS FINAIS — ILPD (seed=123, 80/20)          \n")
cat("=============================================================\n")

cat("\n--- Tabela 1: Treino vs Teste — baseline (thr = 0.50) ---\n")
print(tab_split_base)

cat("\n--- Tabela 2: Treino vs Teste — threshold otimizado (Youden) ---\n")
print(tab_split_thr)

# Determinando o melhor modelo generalista (AUC + Acc) para printar separadamente
todos <- bind_rows(tab_base, tab_thr) %>%
  arrange(desc(AUC), desc(Accuracy), desc(F1))

best_name <- todos$Modelo[1]

cat("\n--- Melhor Modelo ---\n")
cat("Modelo:", best_name, "\n")
cat("AUC:", todos$AUC[1],
    "| Accuracy:", todos$Accuracy[1],
    "| Sensitivity:", todos$Sensitivity[1],
    "| Specificity:", todos$Specificity[1],
    "| F1:", todos$F1[1], "\n")
cat("=============================================================\n")