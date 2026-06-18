# Validacao 232 train_full val_all_3192

## Escopo

- Calcula Cobb regional nas 3192 centerlines `train_full` da validacao completa.
- Nao promove checkpoints e nao altera selectors.
- GT e usado apenas para avaliacao.
- Centerlines: `C:\Users\danib\OneDrive\Ambiente de Trabalho\LPEI\Datasets\Spinal-AI2024-main\Spinal-AI2024-main\processed_padding_512\centerlines\unet_baseline_2000_padding_512_centerline_val_all_3192`.
- Output: `C:\Users\danib\OneDrive\Ambiente de Trabalho\LPEI\Datasets\Spinal-AI2024-main\Spinal-AI2024-main\processed_padding_512\cobb_results\baseline_2000\multicobb\active\validar_cobb_baseline_2000_val_all_3192_v1`.

## Metricas principais

- Max Cobb SMAPE_article: 9.1139
- PT SMAPE_article: 32.1627
- MT SMAPE_article: 10.9728
- TL/L SMAPE_article: 17.7316
- Agg3 SMAPE_article: 17.1345
- MAE3: 4.8443
- RMSE3: 7.104
- <=5: 65.5075%
- <=10: 93.5464%
- <=15: 97.619%
- Falhas >5: 1101
- Severos >8: 358
- Casos completos: 3192
- Falhas de geometria/GT: 0

## Status por target

- MT `ok`: 3192
- PT `ok`: 1830
- PT `ok_secondary_no_upper_curvature`: 1362
- TL_L `ok`: 3099
- TL_L `ok_secondary_low_lower_curvature`: 93

## Leitura

- Esta e a avaliacao absoluta do train_full nas 3192 imagens de validacao.
- Para decidir promocao ainda falta comparar contra uma baseline/checkpoint equivalente nas mesmas 3192 ou aplicar guard em validacao completa.

## Outputs

- `232_case_rows.csv`
- `232_target_rows.csv`
- `232_metrics_summary.csv`
- `232_failures.csv`
- `232_summary.md`
