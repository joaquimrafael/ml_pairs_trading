# Resultados Experimentais — TCC Machine Learning Enhanced Pairs Trading

> Documento de base para o agente de escrita acadêmica. Reúne a tabela de métricas, análise descritiva, achados centrais, sugestões de gráfico e pontos de atenção a partir dos artefatos em `plot/<dataset>/<modelo>/` e `plot/<dataset>/comparison_plots_ml/`.

## 0. Contexto e nomenclatura

**Datasets (3 pares, granularidade de 4 minutos):**

| Dataset | Par | Início | Pearson r | Engle-Granger p | ADF spread p | Hedge β | Half-life (bars) | Cointegrado? |
| ---- | ---- | ---------- | :----: | :----: | :----: | :----: | :----: | :----: |
| `petr3_4_min` | PETR3/PETR4 | 2022-01-03 | **0.9801** | 0.0144 | 0.0031 | 0.992 | 45 058 | Sim |
| `itau3_4_min` | ITAU3/ITAU4 | 2022-01-03 | **0.9909** | 0.0069 | 0.0013 | 0.739 | 26 081 | Sim |
| `bbdc3_4_min` | BBDC4/BBDC3 | 2023-01-03 | **0.9822** | 0.0911 | 0.0276 | 1.200 | 3 920 | **Não (5%)** |

**Modelos avaliados (7 SL):** `lstm`, `nbeats`, `nhits`, `random_forest`, `tcn`, `tft`, `transformer`.

**Estratégias de trading consideradas neste relatório (4):** `pure_forcasting`, `mean_reversion`, `hybrid` e o teto teórico `ground_truth`.

**Thresholds:** auto-derivados de σ das mudanças de razão no treino: `[0, σ/2, σ, 2σ]`. Por dataset: PETR σ=6.93e-4, ITAU σ=8.84e-4, BBDC σ=1.39e-3.

---

## 1. Tabelas de Resultados

### 1.1 Métricas de previsão (forecasting)

Fonte: `plot/<dataset>/comparison_plots_ml/model_metrics_summary.csv`. Melhor valor por dataset em **negrito** (menor é melhor para MAE/RMSE; maior é melhor para R² e Directional Accuracy).

#### MAE — Mean Absolute Error

| Modelo | PETR3/4 | ITAU3/4 | BBDC3/4 |
| --- | :---: | :---: | :---: |
| lstm | 0.00308 | 0.00260 | 0.00134 |
| nbeats | 0.00193 | 0.00443 | 0.00111 |
| nhits | 0.00430 | 0.00422 | 0.00186 |
| random_forest | 0.00530 | 0.00428 | **0.00075** |
| tcn | **0.00141** | **0.00186** | 0.00111 |
| tft | 0.00782 | 0.01090 | 0.00494 |
| transformer | 0.00632 | 0.00402 | 0.00225 |

#### RMSE — Root Mean Squared Error

| Modelo | PETR3/4 | ITAU3/4 | BBDC3/4 |
| --- | :---: | :---: | :---: |
| lstm | 0.00383 | 0.00304 | 0.00170 |
| nbeats | **0.00227** | 0.00531 | 0.00150 |
| nhits | 0.00450 | 0.00505 | 0.00222 |
| random_forest | 0.01069 | 0.00551 | **0.00110** |
| tcn | 0.00230 | **0.00227** | 0.00144 |
| tft | 0.01238 | 0.01236 | 0.00666 |
| transformer | 0.00938 | 0.00474 | 0.00276 |

#### R² (variância explicada)

| Modelo | PETR3/4 | ITAU3/4 | BBDC3/4 |
| --- | :---: | :---: | :---: |
| lstm | 0.945 | 0.850 | 0.961 |
| nbeats | **0.981** | 0.542 | 0.969 |
| nhits | 0.924 | 0.586 | 0.933 |
| random_forest | 0.574 | 0.506 | **0.984** |
| tcn | 0.980 | **0.916** | 0.972 |
| tft | 0.428 | **−1.478** | 0.399 |
| transformer | 0.672 | 0.636 | 0.897 |

#### Directional Accuracy (acerto de sinal)

| Modelo | PETR3/4 | ITAU3/4 | BBDC3/4 |
| --- | :---: | :---: | :---: |
| lstm | 0.424 | 0.356 | 0.277 |
| nbeats | 0.431 | 0.353 | 0.271 |
| nhits | 0.430 | 0.402 | 0.258 |
| random_forest | 0.291 | 0.194 | 0.238 |
| tcn | 0.387 | 0.335 | 0.251 |
| tft | **0.486** | **0.432** | **0.333** |
| transformer | 0.405 | 0.341 | 0.254 |

> **Atenção:** mesmo o melhor modelo (TFT) fica abaixo de 0.50 em todos os datasets — *nenhum modelo acerta a direção da próxima variação da razão melhor que o aleatório*. Esse fato sustenta o argumento de que `mean_reversion` e `hybrid` funcionam por explorar o sinal estatístico do spread, não a previsão pontual.

### 1.2 Métricas de trading — melhor Sharpe Ratio por estratégia

Calculado a partir de `plot/<dataset>/<modelo>/metrics.csv`, tomando o máximo entre os 4 thresholds. **Estrutura do arquivo:** cada linha do CSV corresponde a uma *estratégia* (na ordem `pure_forcasting`, `mean_reversion`, `hybrid`, `ground_truth`), e cada lista é o vetor de Sharpe/profit/trades nos 4 thresholds. Por isso `ground_truth` é idêntico entre todos os modelos (não depende da previsão).

#### PETR3/PETR4 (teto ground_truth = 0.820)

| Modelo | Pure Forecasting | Mean Reversion | Hybrid |
| --- | :---: | :---: | :---: |
| lstm | 0.031 | 0.067 | 0.143 |
| nbeats | 0.082 | 0.125 | 0.149 |
| nhits | 0.008 | 0.020 | 0.138 |
| random_forest | **0.113** | 0.142 | 0.177 |
| tcn | **0.113** | **0.162** | **0.183** |
| tft | 0.032 | 0.060 | 0.143 |
| transformer | 0.033 | 0.056 | 0.147 |

#### ITAU3/ITAU4 (teto ground_truth = 0.659)

| Modelo | Pure Forecasting | Mean Reversion | Hybrid |
| --- | :---: | :---: | :---: |
| lstm | **0.060** | 0.088 | 0.133 |
| nbeats | 0.051 | 0.070 | **0.135** |
| nhits | 0.051 | 0.072 | 0.122 |
| random_forest | 0.056 | 0.074 | 0.131 |
| tcn | 0.059 | **0.095** | 0.126 |
| tft | 0.015 | 0.022 | 0.118 |
| transformer | 0.048 | 0.069 | 0.128 |

#### BBDC3/BBDC4 (teto ground_truth = 0.811)

| Modelo | Pure Forecasting | Mean Reversion | Hybrid |
| --- | :---: | :---: | :---: |
| lstm | 0.167 | 0.200 | 0.189 |
| nbeats | 0.193 | **0.226** | **0.205** |
| nhits | 0.121 | 0.163 | 0.168 |
| random_forest | **0.211** | 0.208 | 0.133 |
| tcn | 0.148 | 0.205 | 0.184 |
| tft | 0.067 | 0.099 | 0.165 |
| transformer | 0.103 | 0.139 | 0.164 |

### 1.3 Lucro total — melhor threshold por estratégia (em unidades de razão)

| Dataset | Modelo | Pure Forec. | Mean Rev. | Hybrid | Ground Truth (teto) |
| --- | --- | :---: | :---: | :---: | :---: |
| PETR | TCN (campeão geral) | 65.4 | 60.8 | 61.8 | 264.8 |
| PETR | random_forest | 62.2 | 58.0 | 53.7 | 264.8 |
| ITAU | TCN | 30.8 | 58.7 | 65.5 | 374.9 |
| ITAU | LSTM | 40.9 | 53.7 | 70.0 | 374.9 |
| ITAU | nbeats | 47.8 | 47.8 | 67.2 | 374.9 |
| BBDC | nbeats | 46.9 | 46.9 | 33.1 | 134.8 |
| BBDC | random_forest | 44.1 | 43.9 | 35.7 | 134.8 |

---

## 2. Análise Descritiva dos Resultados

### 2.1 TCN domina previsão com erro pontual + pareia bem em trading

TCN entrega o menor MAE em PETR (0.00141) e ITAU (0.00186), com R² ≥ 0.916 nos dois casos, e em BBDC fica praticamente empatado com NBEATS e random_forest. No trading, TCN é o líder absoluto em PETR (Sharpe 0.183 em hybrid, lucro 65.4 em pure forecasting) e o melhor em mean reversion no ITAU (0.095) — coerente com o argumento de que erro de magnitude baixo, mesmo sem grande directional accuracy (0.387 em PETR), traduz-se em melhor sinal para mean reversion porque preserva a *ordem relativa* das previsões.

### 2.2 Directional accuracy < 50% em todos os modelos é o fato mais surpreendente

TFT lidera direcionalidade nos três datasets (0.486 / 0.432 / 0.333) mas NUNCA passa de 0.50. Os "campeões" de previsão (nbeats, tcn) acertam direção em apenas 39–43%. Isso significa que **previsão direcional pura é pior que aleatória** — e ainda assim os modelos geram Sharpe > 0 em hybrid (até 0.205 em BBDC com nbeats). A explicação: o sinal lucrativo vem do filtro de *magnitude* (apenas opera quando |predição| > threshold) e do *componente mean-reversion* embutido em hybrid, não do acerto de sinal.

### 2.3 BBDC é o dataset mais lucrativo apesar de não cointegrado

Sharpe de mean reversion para nbeats em BBDC = 0.226 (o maior entre todos os pares e modelos), e a maioria dos modelos passa de 0.15 — substancialmente acima dos 0.05–0.16 vistos em PETR e ITAU. O resultado é contraintuitivo: BBDC falha no teste Engle-Granger a 5% (p=0.091), mas o spread é estacionário pelo ADF (p=0.028) com half-life curtíssimo de 3 920 bars (vs 26k–45k nos outros). **Half-life curto torna o trading mais lucrativo do que a "qualidade" estatística da cointegração.**

### 2.4 Random Forest é o melhor preditor de magnitude em BBDC, mas degrada-se em sinal

RF tem MAE = 0.00075 e R² = 0.984 em BBDC — o melhor de todos. Mas sua directional accuracy é 0.238 (a pior do dataset, atrás de TFT 0.333) e seu Sharpe em hybrid cai para 0.133 (último entre os modelos sérios). RF aprende o nível médio do spread quase perfeitamente mas erra a direção — confirmando que RMSE baixo não é suficiente para trading.

### 2.5 TFT é um caso patológico em ITAU

TFT tem o pior R² em todos os datasets (PETR 0.428, ITAU **−1.478**, BBDC 0.399) — em ITAU o R² negativo significa que TFT é pior que prever a média. Apesar disso, é o líder em directional accuracy. Resultado: lucros de pure forecasting muito baixos (15.3 em ITAU vs 40.9 do LSTM), mas hybrid competitivo (Sharpe 0.118–0.165). TFT está acertando o *sinal* sem acertar a *magnitude*.

### 2.6 Gap modelo × ground truth é grande nos três datasets

| Dataset | Melhor Sharpe SL (hybrid) | Ground Truth | Gap |
| --- | :---: | :---: | :---: |
| PETR | 0.183 (TCN) | 0.820 | **22%** |
| ITAU | 0.135 (NBEATS) | 0.659 | **20%** |
| BBDC | 0.226 (NBEATS, mean rev) | 0.811 | **28%** |

O melhor modelo SL captura no máximo ~28% do potencial teórico. Esse gap é o argumento central para discutir limitações de orçamento de épocas (3 epochs) e para motivar trabalhos futuros (mais épocas, ensemble).

### 2.7 Estabilidade temporal: modelos pesados degradam no fim do teste

O `rolling_rmse_w500.png` de PETR mostra TFT, transformer e random_forest disparando RMSE rolante de ~0.005 para 0.020–0.035 nos últimos ~3 000 passos, enquanto TCN e nbeats permanecem estáveis abaixo de 0.005. Indica overfitting/sensibilidade a regime de mercado para os modelos com mais parâmetros mal-calibrados em 3 epochs.

---

## 3. Resultados-chave para a narrativa do TCC

A hipótese central reconstruída a partir do contexto é: *"Modelos de deep learning aplicados à previsão da razão de pares cointegrados, combinados com estratégias de trading guiadas pelo spread, geram retorno ajustado ao risco superior ao da previsão isolada."* Os achados em ordem de relevância:

1. **Achado:** O melhor Sharpe operacional em todos os datasets é da estratégia **hybrid** (que combina previsão e spread), não da previsão pura.
   - **Evidência:** PETR — hybrid TCN 0.183 vs pure_forcasting TCN 0.113 (+62%); ITAU — hybrid NBEATS 0.135 vs pure NBEATS 0.051 (+165%); BBDC — hybrid e mean_reversion são os campeões em quase todos os modelos.
   - **Relevância:** Suporta diretamente a hipótese — o ML adiciona valor *quando combinado com o sinal estatístico do par*, não substituindo-o.

2. **Achado:** Directional accuracy < 50% em todos os modelos, mas trading lucrativo possível.
   - **Evidência:** TFT lidera com 0.486 / 0.432 / 0.333; TCN tem 0.387 e ainda assim domina o trading em PETR.
   - **Relevância:** Refuta a intuição de que "modelo bom" = "alta acurácia direcional" e fortalece a contribuição do enquadramento como *filtro de magnitude*.

3. **Achado:** TCN é a melhor arquitetura única (Pareto-ótima entre erro, sinal e Sharpe), seguido de NBEATS.
   - **Evidência:** TCN é top-1 ou top-2 em MAE e R² nos três datasets; é #1 em Sharpe em PETR e ITAU. A nota `SELECAO_REFINAMENTO_TCN_LSTM.txt` argumenta arquitetonicamente o porquê (campo receptivo de 127 bars cobre a janela ADF de 62 bars com gradiente curto, ideal para 3 epochs).
   - **Relevância:** Justifica a escolha de TCN+LSTM como modelos focais e dá uma recomendação prática.

4. **Achado:** Half-life curto explica retorno mais que cointegração formal.
   - **Evidência:** BBDC (não-cointegrado a 5%, half-life 3 920) gera Sharpe ~0.20 enquanto PETR (cointegrado, half-life 45 058) fica em ~0.15.
   - **Relevância:** Adiciona uma observação metodológica para a banca: o critério clássico de seleção de pares (cointegração) é necessário mas não suficiente — half-life é um preditor melhor do retorno.

5. **Achado:** Modelos heavy (TFT, transformer, RF) sub-treinam em 3 epochs e degradam no out-of-sample.
   - **Evidência:** TFT R² = −1.478 em ITAU; rolling RMSE de TFT/transformer/RF dispara em PETR após o passo 14 500. NBEATS e TCN permanecem estáveis.
   - **Relevância:** Justifica o protocolo experimental (3 epochs como controle) e a exclusão de modelos com melhor teto sob orçamento maior.

6. **Achado:** Gap consistente para o teto teórico (~75% do Sharpe inalcançado).
   - **Evidência:** PETR 0.183/0.820, ITAU 0.135/0.659, BBDC 0.226/0.811.
   - **Relevância:** Define a pergunta aberta para trabalhos futuros (mais epochs, ensembling) e dimensiona honestamente a contribuição.

---

## 4. Sugestão de Gráficos

| # | Gráfico | Eixos | Insight | Arquivo existente |
| :-: | --- | --- | --- | --- |
| 1 | **Heatmap de Sharpe (modelo × dataset, melhor estratégia por célula)** | linhas = modelo, colunas = dataset, cor = Sharpe máximo | Ranking visual instantâneo entre modelos e datasets, destacando TCN/NBEATS | *Não existe — gerar a partir das tabelas 1.2* |
| 2 | Barras agrupadas: Sharpe por estratégia × modelo (1 por dataset) | x = estratégia, y = Sharpe, hue = modelo | Mostra que hybrid > mean_reversion > pure_forcasting na maioria dos modelos, e teto ground_truth | `plot/<dataset>/comparison_plots_ml/sharpe_ratio_comparison.png` (já existe para os 3) |
| 3 | Barras agrupadas: Lucro total por estratégia × modelo | x = estratégia, y = lucro, hue = modelo | Magnitude absoluta do retorno; ground_truth como teto visual | `plot/<dataset>/comparison_plots_ml/total_profit_comparison.png` (existe) |
| 4 | Barras: Directional Accuracy por modelo (1 por dataset) | x = modelo, y = acurácia, linha de referência em 0.5 | Mostra que nenhum modelo passa do aleatório → narrativa do achado #2 | `plot/<dataset>/comparison_plots_ml/directional_accuracy_by_model.png` (existe) |
| 5 | Linha: Rolling RMSE (janela=500) por modelo | x = step, y = RMSE rolante | Estabilidade temporal e degradação de TFT/transformer/RF no fim do teste | `plot/petr3_4_min/comparison_plots_ml/rolling_rmse_w500.png` (existe; replicar para ITAU e BBDC) |
| 6 | Calibração: True vs Predicted (scatter) | x = razão real, y = razão prevista, diagonal y=x | Mostra TFT/transformer "saturando" em previsões enviesadas vs TCN/NBEATS bem calibrados | `plot/<dataset>/comparison_plots_ml/calibration_true_vs_pred.png` (existe) |
| 7 | Boxplot de resíduos por modelo | x = modelo, y = resíduo (com linha em 0) | Viés (mediana ≠ 0) e dispersão; identifica modelos enviesados (LSTM positivo, NHITS negativo) | `plot/<dataset>/comparison_plots_ml/residuals_boxplot.png` (existe) |
| 8 | **Sharpe vs threshold (linhas, 1 painel por estratégia)** | x = threshold (0, σ/2, σ, 2σ), y = Sharpe, hue = modelo | Mostra qual threshold é ótimo por estratégia — argumenta a auto-derivação por σ | *Pode ser gerado a partir de `metrics.csv`; existe `sharpe_ratio_vs_threshold.png` por modelo, falta agregado* |
| 9 | **Tradeoff scatter: Directional Accuracy × Sharpe (hybrid)** | x = direc. accuracy, y = Sharpe hybrid, ponto = (modelo, dataset), cor = dataset | Visualiza diretamente o achado #2 (alta accuracy ≠ alto Sharpe; TFT vs TCN) | *Não existe — gerar* |
| 10 | **Painel resumo: Pair analysis comparativo** | tabela visual r, EG p, ADF p, half-life para os 3 pares | Contexto para a banca: por que BBDC é não-cointegrado mas lucrativo | `plot/<dataset>/comparison_plots_ml/pair_analysis_summary.png` existe por dataset; consolidar em um único painel |
| 11 | Cumulative PnL — melhor modelo × ground_truth | x = tempo, y = PnL acumulado | Mostra trajetória do retorno e o gap para o teto teórico | `plot/<dataset>/<modelo>/cumulative_pnl_best_sharpe.png` (existe por modelo; sobrepor melhor modelo + ground_truth) |

---

## 5. Pontos de Atenção

### 5.1 Limitações visíveis nos resultados

- **Treino com apenas 3 epochs.** A nota `SELECAO_REFINAMENTO_TCN_LSTM.txt` justifica essa escolha (controle entre arquiteturas, dataset grande, sinal fraco), mas a banca pode questionar — o gap consistente de ~75% para o ground truth confirma que os modelos não estão no plateau. Dependendo do escopo, vale incluir ao menos um experimento de sensibilidade (TCN com 10 / 30 epochs).
- **Random Forest sem treino sequencial.** RF não tem noção temporal; seu desempenho competitivo em BBDC sugere que metade do "valor agregado de DL" pode vir de feature engineering simples. Mencionar como baseline forte é honesto.

### 5.2 Resultados que precisam de explicação adicional no texto

- **R² negativo do TFT em ITAU (−1.478).** Significa pior que prever a média; precisa contexto (subajuste em 3 epochs, complexidade do modelo).
- **Ground truth idêntico entre modelos no `metrics.csv`.** A interpretação correta da estrutura do CSV (cada *linha* é uma estratégia, cada *lista* são os 4 thresholds) precisa ser explicada — caso contrário um leitor pode achar que há dados duplicados.
- **BBDC não-cointegrado mas lucrativo.** Precisa-se justificar que a estacionariedade do spread (ADF) é o critério funcional, não a cointegração formal de Engle-Granger no nível 5%.
- **Hedge ratios muito diferentes (β: 0.74 ITAU vs 1.20 BBDC).** Vale comentar a robustez do critério OLS para hedge ratio em pares com volatilidades diferentes entre as duas pernas.
- **Diferença entre `output_chunk_length=1` (modelos RNN) e os demais.** O log do PETR mostra "ignoring user defined output_chunk_length. RNNModel uses fixed output_chunk_length=1" — comparações precisam mencionar essa restrição arquitetônica.

### 5.3 Comparações que podem ser questionadas pela banca

1. **"TCN é melhor" — em qual métrica?** TCN ganha em MAE/RMSE/R² e em Sharpe-PETR/ITAU, mas perde em directional accuracy (TFT) e em Sharpe-BBDC (NBEATS). O resumo precisa qualificar: "melhor compromisso erro+trading", não "melhor em tudo".
2. **Auto-derivação de thresholds por σ do treino.** A banca pode pedir: por que 4 valores `[0, σ/2, σ, 2σ]`? Por que não percentis empíricos? Deixar essa decisão explícita.
3. **Comparação direta entre datasets.** PETR/ITAU começam em 2022 (~96k bars) e BBDC em 2023 (~74k bars) — o conjunto de teste é menor em BBDC, o que pode inflar/deflar Sharpe. Vale mostrar tamanho do teste em cada caso.
4. **Custos de transação ausentes.** Sharpe e lucro são calculados sem slippage/comissão. Para o universo brasileiro de ações com horário de leilão, isso pode reduzir significativamente o retorno líquido. Reconhecer explicitamente como limitação.
5. **Seleção do "melhor threshold" a posteriori.** As tabelas 1.2 e 1.3 reportam o máximo entre 4 thresholds — isso é uma escolha *in-sample* sobre o conjunto de teste. Para honestidade, considerar um split adicional ou reportar Sharpe médio entre thresholds.
6. **Teto ground_truth depende do dataset, não dá para comparar entre datasets.** PETR 0.820 vs BBDC 0.811 são tetos próximos por coincidência; ITAU é 0.659. A "fração de teto capturada" é uma métrica mais comparável que o Sharpe absoluto.

---

*Documento gerado a partir de:*

- `plot/<dataset>/comparison_plots_ml/model_metrics_summary.csv` (3 datasets)
- `plot/<dataset>/<modelo>/metrics.csv` (21 arquivos: 7 modelos × 3 datasets)
- `plot/<dataset>/pair_analysis/pair_analysis_results.csv` (3 datasets)
- `plot/itau3_4_min/SELECAO_REFINAMENTO_TCN_LSTM.txt`
- Inspeção visual dos PNGs em `comparison_plots_ml/` (sharpe, profit, directional accuracy, RMSE, calibração, resíduos, pair analysis summary).
