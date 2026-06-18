# Validacao 232 train_full val_all_3192

## Escopo

- Calcula Cobb regional nas 3192 centerlines `train_full` da validacao completa.
- Nao promove checkpoints e nao altera selectors.
- GT e usado apenas para avaliacao.
- Centerlines: `C:\Users\danib\OneDrive\Ambiente de Trabalho\LPEI\Datasets\Spinal-AI2024-main\Spinal-AI2024-main\processed_padding_512\centerlines\unet_train_full_padding_512_centerline_val_all_3192`.
- Output: `C:\Users\danib\OneDrive\Ambiente de Trabalho\LPEI\Datasets\Spinal-AI2024-main\Spinal-AI2024-main\processed_padding_512\cobb_results\train_full\multicobb\active\validar_cobb_trainfull_val_all_3192_v1`.

## Metricas principais

- Max Cobb SMAPE_article: 9.2067
- PT SMAPE_article: 32.5679
- MT SMAPE_article: 11.0812
- TL/L SMAPE_article: 17.9422
- Agg3 SMAPE_article: 17.3259
- MAE3: 4.8881
- RMSE3: 7.1302
- <=5: 65.0689%
- <=10: 93.5777%
- <=15: 97.619%
- Falhas >5: 1115
- Severos >8: 359
- Casos completos: 3192
- Falhas de geometria/GT: 0

## Status por target

- MT `ok`: 3192
- PT `ok`: 1783
- PT `ok_secondary_no_upper_curvature`: 1409
- TL_L `ok`: 3105
- TL_L `ok_secondary_low_lower_curvature`: 87

## Leitura

- Esta e a avaliacao absoluta do train_full nas 3192 imagens de validacao.
- Para decidir promocao ainda falta comparar contra uma baseline/checkpoint equivalente nas mesmas 3192 ou aplicar guard em validacao completa.

## Outputs

- `232_case_rows.csv`
- `232_target_rows.csv`
- `232_metrics_summary.csv`
- `232_failures.csv`
- `232_summary.md`
