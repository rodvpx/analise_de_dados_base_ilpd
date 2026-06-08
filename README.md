# Analise de Dados - Indian Liver Patient Dataset (ILPD)

Este projeto aplica um pipeline de analise de dados e machine learning sobre o **Indian Liver Patient Dataset (ILPD)**, com o objetivo de prever a presenca de doenca hepatica a partir de atributos clinicos e laboratoriais.

A abordagem final combinou pre-processamento, engenharia de variaveis medicas, selecao de atributos, validacao cruzada, modelos supervisionados e ajuste de threshold para melhorar a tomada de decisao no conjunto de teste.

## Dataset

O arquivo utilizado foi:

- `Indian Liver Patient Dataset (ILPD).csv`

A base original possui **583 registros** e **11 colunas**. A variavel alvo original, `Selector`, foi convertida para classificacao binaria:

- `Yes`: paciente com doenca hepatica
- `No`: paciente sem doenca hepatica

Distribuicao original da classe alvo:

| Classe original | Interpretacao | Registros |
|---|---:|---:|
| 1 | Doenca hepatica | 416 |
| 2 | Sem doenca hepatica | 167 |

A base e desbalanceada, com predominio de pacientes classificados como doentes.

## Variaveis Originais

As colunas foram renomeadas no script para facilitar a leitura:

| Coluna | Descricao |
|---|---|
| `Age` | Idade |
| `Gender` | Genero |
| `TB` | Bilirrubina total |
| `DB` | Bilirrubina direta |
| `Alkphos` | Fosfatase alcalina |
| `SGPT` | Alanina aminotransferase |
| `SGOT` | Aspartato aminotransferase |
| `TP` | Proteinas totais |
| `ALB` | Albumina |
| `AG_Ratio` | Razao albumina/globulina |
| `Selector` | Classe alvo |

## Pipeline Utilizado

O pipeline principal esta implementado em `script.r`.

Etapas executadas:

1. Carregamento da base ILPD.
2. Remocao de linhas duplicadas.
3. Conversao de `Gender` para formato numerico.
4. Imputacao de valores ausentes em `AG_Ratio` pela mediana.
5. Conversao da variavel alvo para fator binario (`No`, `Yes`).
6. Criacao de novas variaveis clinicas e transformacoes logaritmicas.
7. Separacao treino/teste estratificada em proporcao 80/20 com `seed = 123`.
8. Remocao de variaveis com baixa variancia e alta correlacao.
9. Selecao de variaveis com StepAIC backward.
10. Normalizacao feita apenas com os dados de treino para evitar vazamento de dados.
11. Treinamento com validacao cruzada repetida: 10 folds, 5 repeticoes.
12. Tratamento do desbalanceamento por pesos de classe.
13. Treinamento e comparacao de multiplos modelos.
14. Calibracao do XGBoost com Platt Scaling.
15. Ensemble por stacking ponderado por AUC.
16. Otimizacao de threshold pelo indice de Youden.
17. Geracao de graficos de ROC, importancia de variaveis, probabilidades, matriz de confusao e comparacao por AUC.

## Feature Engineering

Foram criadas variaveis adicionais com base em relacoes clinicas comuns em exames hepaticos:

| Variavel criada | Ideia |
|---|---|
| `AST_ALT_ratio` | Razao entre SGOT e SGPT |
| `Bilirubin_ratio` | Razao entre bilirrubina direta e total |
| `Protein_ratio` | Razao entre albumina e proteinas totais |
| `TB_DB_ratio` | Relacao entre bilirrubina total e direta |
| `SGPT_SGOT_ratio` | Relacao inversa entre SGPT e SGOT |
| `Indirect_Bili` | Bilirrubina indireta estimada |
| `Globulin` | Globulina estimada por `TP - ALB` |
| `log_SGPT`, `log_SGOT`, `log_Alkphos`, `log_TB`, `log_DB` | Transformacoes `log1p` para reduzir assimetria |

## Modelos Avaliados

Foram avaliados os seguintes modelos:

| Modelo | Observacao |
|---|---|
| Random Forest | Treinado com pesos de classe |
| Elastic Net | Modelo regularizado com busca em grade para `alpha` e `lambda` |
| Regressao Logistica | Baseline interpretavel |
| SVM Radial | Kernel radial com busca em grade para `sigma` e `C` |
| XGBoost gbtree | Busca em grade, validacao cruzada e early stopping |
| Ensemble Stack | Combinacao ponderada das probabilidades dos modelos por AUC |

## Resultados Obtidos

Os resultados abaixo foram registrados em `resultado.txt`, usando divisao treino/teste 80/20 com `seed = 123`.

### Comparacao com Threshold Padrao 0.50

| Modelo | Train Acc | Train AUC | Test Acc | Test AUC | Test Sens | Test Spec | Test F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| XGBoost gbtree | 0.812 | 0.870 | 0.681 | 0.744 | 0.827 | 0.312 | 0.788 |
| Ensemble Stack | 0.851 | 0.938 | 0.681 | 0.742 | 0.889 | 0.156 | 0.800 |
| Elastic Net | 0.742 | 0.775 | 0.681 | 0.732 | 0.877 | 0.188 | 0.798 |
| Regressao Logistica | 0.737 | 0.774 | 0.673 | 0.730 | 0.864 | 0.188 | 0.791 |
| Random Forest | 1.000 | 1.000 | 0.708 | 0.720 | 0.901 | 0.219 | 0.816 |
| SVM Radial | 0.716 | 0.784 | 0.708 | 0.693 | 0.988 | 0.000 | 0.829 |

### Comparacao com Threshold Otimizado por Youden

| Modelo | Threshold | Train Acc | Train AUC | Test Acc | Test AUC | Test Sens | Test Spec | Test F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| XGBoost gbtree | 0.601 | 0.825 | 0.870 | 0.717 | 0.744 | 0.802 | 0.500 | 0.802 |
| Ensemble Stack | 0.604 | 0.891 | 0.938 | 0.690 | 0.742 | 0.815 | 0.375 | 0.790 |
| Elastic Net | 0.769 | 0.648 | 0.775 | 0.611 | 0.732 | 0.519 | 0.844 | 0.656 |
| Regressao Logistica | 0.725 | 0.676 | 0.774 | 0.655 | 0.730 | 0.642 | 0.688 | 0.727 |
| Random Forest | 0.488 | 1.000 | 1.000 | 0.717 | 0.720 | 0.914 | 0.219 | 0.822 |
| SVM Radial | 0.721 | 0.713 | 0.784 | 0.646 | 0.693 | 0.654 | 0.625 | 0.726 |

## Melhor Resultado

O melhor modelo consolidado foi:

| Modelo | AUC | Accuracy | Sensitivity | Specificity | F1 |
|---|---:|---:|---:|---:|---:|
| XGBoost gbtree com threshold otimizado | 0.744 | 0.717 | 0.802 | 0.500 | 0.802 |

Esse resultado indica que o XGBoost foi o modelo com melhor equilibrio geral no teste, principalmente por manter a maior AUC e melhorar a especificidade apos o ajuste do threshold.

## Interpretacao dos Resultados

O desempenho final mostra que o problema e dificil por tres motivos principais:

- A base e pequena, com apenas 583 registros originais.
- As classes sao desbalanceadas, com maior quantidade de pacientes doentes.
- Algumas variaveis laboratoriais possuem distribuicoes assimetricas e relacoes fortes entre si.

Pontos importantes observados:

- O Random Forest atingiu AUC e acuracia perfeitas no treino, mas caiu no teste, indicando overfitting.
- O SVM com threshold 0.50 teve sensibilidade muito alta, mas especificidade igual a zero, ou seja, praticamente classificou todos como positivos.
- O ajuste de threshold pelo indice de Youden melhorou o equilibrio entre sensibilidade e especificidade em varios modelos.
- O XGBoost apresentou a melhor AUC no teste, com desempenho mais equilibrado apos threshold otimizado.
- O Ensemble Stack ficou muito proximo do XGBoost em AUC, mas nao superou o modelo individual no resultado final.

## Graficos Gerados

O script gera os seguintes graficos durante a execucao:

- Curvas ROC comparando todos os modelos.
- Importancia das variaveis no XGBoost.
- Distribuicao das probabilidades previstas pelo XGBoost.
- Matriz de confusao do XGBoost com threshold otimizado.
- Comparacao de AUC entre modelos.
- Status das variaveis apos selecao StepAIC.

## Estrutura do Projeto

```text
.
├── Indian Liver Patient Dataset (ILPD).csv
├── README.md
├── resultado.txt
├── script.r
└── ilpd+indian+liver+patient+dataset.Rproj
```

## Arquivos Principais

| Arquivo | Funcao |
|---|---|
| `script.r` | Pipeline completo de analise, treinamento, avaliacao e graficos |
| `resultado.txt` | Saida registrada com as metricas finais |
| `Indian Liver Patient Dataset (ILPD).csv` | Base de dados utilizada |
| `README.md` | Documentacao do projeto |

## Conclusao

O projeto conseguiu construir um pipeline completo e reprodutivel para classificacao de pacientes no ILPD. Entre os modelos testados, o **XGBoost gbtree com threshold otimizado** apresentou o melhor resultado geral no conjunto de teste, com **AUC de 0.744**, **acuracia de 0.717** e **F1 de 0.802**.

Apesar do resultado ser consistente para uma base pequena e desbalanceada, o desempenho ainda deve ser interpretado com cuidado. Para uso real, seria necessario validar o modelo em uma base externa, revisar custos clinicos de falsos negativos e falsos positivos, e ajustar o threshold de acordo com o objetivo medico do sistema.
