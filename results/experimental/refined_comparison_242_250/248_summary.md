# Resumo 248

## Estado

- Pipeline fixa aplicada com candidatos regenerados.
- Sem promocao automatica.

## Metricas finais

### train2000
- Max Cobb SMAPE_article: 8.3902
- PT SMAPE_article: 31.0149
- MT SMAPE_article: 9.8407
- TL/L SMAPE_article: 17.955
- Agg3 SMAPE_article: 16.5162
- MAE3: 4.7172
- RMSE3: 7.1646
- <=5: 69.4549%
- <=10: 93.4211%
- <=15: 97.4624%
- falhas >5: 975
- severos >8: 310

### trainfull
- Max Cobb SMAPE_article: 8.4957
- PT SMAPE_article: 31.4553
- MT SMAPE_article: 9.9073
- TL/L SMAPE_article: 18.2135
- Agg3 SMAPE_article: 16.7131
- MAE3: 4.7653
- RMSE3: 7.2134
- <=5: 68.8283%
- <=10: 93.2644%
- <=15: 97.3371%
- falhas >5: 995
- severos >8: 310

## Comparacao

- Delta trainfull - train2000 em <=5: -0.6266 pontos.
- Delta trainfull - train2000 em MAE3: 0.0481.

## Leitura

- A pergunta respondida aqui e o efeito da centerline train_full mantendo a mesma composicao congelada de candidatos/perfis.
- A interpretacao deve considerar as limitacoes registadas no `248_manifest.md`.
