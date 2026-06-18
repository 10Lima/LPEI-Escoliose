# Auditoria 250 - paridade checkpoint205 vs 248 nos 1000 historicos

## Primeira divergencia

- Primeira etapa divergente: `BASE186` com 990/1000 casos divergentes.
- Delta <=5 nessa etapa: -17.9 pontos.
- Delta MAE3 nessa etapa: 1.2449.

## Etapas

- BASE186: divergentes 990/1000, <=5 hist 80.9 vs recon 63.0, MAE3 hist 3.7169 vs recon 4.9618.
- MT190: divergentes 983/1000, <=5 hist 81.7 vs recon 67.1, MAE3 hist 3.6893 vs recon 4.8406.
- PT193: divergentes 983/1000, <=5 hist 82.1 vs recon 67.6, MAE3 hist 3.6699 vs recon 4.8131.
- TLL197: divergentes 983/1000, <=5 hist 82.3 vs recon 68.2, MAE3 hist 3.656 vs recon 4.786.
- PT200: divergentes 982/1000, <=5 hist 82.4 vs recon 68.2, MAE3 hist 3.6261 vs recon 4.786.
- SF204: divergentes 981/1000, <=5 hist 82.4 vs recon 68.2, MAE3 hist 3.6247 vs recon 4.8219.

## 10 artefactos mais importantes em falta/nao equivalentes

- 1. `selector125_oof_candidate_predictions.csv`: not_equivalent. O regenerado 245 nao contem predicted_good_score OOF equivalente ao selector125; usa score geometrico/selection_score.
- 2. `gate123_oof_candidate_predictions.csv`: not_equivalent. O regenerado 244 nao contem gate123_score/predicted_good_score OOF equivalentes; PT perde a semantica do gate historico.
- 3. `gate114_oof_candidate_predictions.csv`: not_equivalent. O regenerado 246 nao separa o gate antigo old_TLL/gate114 com score OOF equivalente.
- 4. `tll159_candidate_rows.csv`: partial. O 246 tem candidatos TL/L novos, mas nao reproduz exatamente tll159 source/method/rank.
- 5. `tll175_candidate_rows.csv`: partial. O 246 tem variantes TL/L, mas nao reproduz exatamente tll175 source/method/rank.
- 6. `prep203_candidate_rows.csv`: not_equivalent. O 247 e derivado dos candidatos 244/246; nao reproduz prep203 construido a partir de PT193/TLL196 historicos.
- 7. `sf204_best_safe_selected_cases.csv`: not_equivalent. O overlay severe do 248 nao tem o mesmo universo/foco de 25 severos do checkpoint201 historico.
- 8. `selector186_best_target_rows.csv`: not_equivalent. A base do 248 vem de Cobb direto por centerline/case summaries; nao materializa o checkpoint186 historico.
- 9. `diagnostico192_target_rows.csv`: missing. Nao existe diagnostico192 equivalente por CENTERLINE_SUBDIR para definir foco PT dominante/falhado sem recorrer ao historico.
- 10. `diagnostico195_target_rows.csv`: missing. Nao existe diagnostico195 equivalente por CENTERLINE_SUBDIR para reproduzir foco TL/L historico.

## Resposta

Para que o 248 train2000 reproduza o checkpoint205 historico, faltam equivalentes fieis dos artefactos de candidatos, gates OOF e diagnosticos de foco usados pela cadeia 190-205.
A primeira divergencia deve ser tratada antes de qualquer tentativa de melhorar metricas.
