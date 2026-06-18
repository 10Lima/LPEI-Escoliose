# Auditoria 249 - 248 nos 1000 do checkpoint205 vs restantes

## Cruzamento

- Raw IDs no checkpoint205 historico: 1000.
- Encontrados no 248 por raw_file_id: 1000.
- Ausentes no 248: 0.

## Metricas principais

### checkpoint205_historico / checkpoint205_overlap_1000
- casos: 1000
- <=5: 82.4%
- MAE3: 3.6247
- RMSE3: 4.7629
- falhas >5: 176
- severos >8: 24

### train2000_248 / checkpoint205_overlap_1000
- casos: 1000
- <=5: 68.2%
- MAE3: 4.8219
- RMSE3: 7.3832
- falhas >5: 318
- severos >8: 104

### train2000_248 / remaining_2192
- casos: 2192
- <=5: 70.0274%
- MAE3: 4.6694
- RMSE3: 7.0626
- falhas >5: 657
- severos >8: 206

### train2000_248 / all_3192
- casos: 3192
- <=5: 69.4549%
- MAE3: 4.7172
- RMSE3: 7.1646
- falhas >5: 975
- severos >8: 310

### trainfull_248 / checkpoint205_overlap_1000
- casos: 1000
- <=5: 67.4%
- MAE3: 4.8451
- RMSE3: 7.4417
- falhas >5: 326
- severos >8: 96

### trainfull_248 / remaining_2192
- casos: 2192
- <=5: 69.4799%
- MAE3: 4.7288
- RMSE3: 7.1069
- falhas >5: 669
- severos >8: 214

### trainfull_248 / all_3192
- casos: 3192
- <=5: 68.8283%
- MAE3: 4.7653
- RMSE3: 7.2134
- falhas >5: 995
- severos >8: 310

## Leitura

- Se o 248 nos 1000 do checkpoint205 ficar longe do checkpoint205 historico, o problema principal esta nos candidatos/gates regenerados.
- Se o 248 nos 1000 ficar perto e cair nos restantes 2192, o problema principal e generalizacao para val_all_3192.
