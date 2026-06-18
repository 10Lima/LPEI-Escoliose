# Checkpoint205 checkpoint201 + SF204

## Entradas

- Checkpoint201: `C:\Users\danib\OneDrive\Ambiente de Trabalho\LPEI\Datasets\Spinal-AI2024-main\Spinal-AI2024-main\processed_padding_512\cobb_results\baseline_2000\multicobb\active\checkpoint201_checkpoint198_pt200_v1`.
- SF204 validado: `C:\Users\danib\OneDrive\Ambiente de Trabalho\LPEI\Datasets\Spinal-AI2024-main\Spinal-AI2024-main\processed_padding_512\cobb_results\baseline_2000\multicobb\active\validar_severe_first_pt_tll_checkpoint201_v1`.

## Selecoes

- Substituicoes severe-first seguras em 2 targets.
- MT mantido do checkpoint201.

## Metricas principais

| Stage | Max Cobb SMAPE_article | PT | MT | TL/L | Agg3 | MAE3 | RMSE3 | <=5 | Falhas | Severos | Reg target >5 | Reg img >1 | Reg img >3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| checkpoint201 | 8.0634 | 24.5901 | 8.6292 | 12.4393 | 12.8402 | 3.6261 | 4.7658 | 82.4 | 176 | 25 | 0 | 0 | 0 |
| checkpoint205 | 8.062 | 24.5901 | 8.6292 | 12.427 | 12.8357 | 3.6247 | 4.7629 | 82.4 | 176 | 24 | 0 | 0 | 0 |

## Leitura

- Falhas: 176 -> 176.
- Severos: 25 -> 24.
- MAE3: 3.6261 -> 3.6247.
- <=5: 82.4 -> 82.4.
- Este script promove o candidato seguro validado no 204 para um checkpoint reprodutivel.
