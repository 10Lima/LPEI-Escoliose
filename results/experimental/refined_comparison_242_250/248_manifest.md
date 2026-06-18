# Manifest 248

## Objetivo

Aplicar a pipeline refinada final com candidatos regenerados por centerline e comparar train2000 vs trainfull.

## Regras

- Usa apenas candidatos regenerados pelos scripts 244, 245, 246 e 247.
- Nao promove checkpoint.
- Severe-first e overlay seletivo: se nao houver candidato, mantem a etapa anterior.
- Nao reoptimiza configs com GT.
- Replay diagnostico de gates historicos com colunas GT: `1`.

## Perfis congelados usados

- MT190: `{'config_id': 'mt190__baseerr_4.5__posjump_3.6__rank_500__jump_4__score_0.5__failorerr_0__passmargin_0.5', 'max_rank': 500, 'max_jump': 4.0, 'min_score': 0.5, 'max_positive_jump': 3.6, 'min_base_error_diag': 4.5}`.
- PT193: `{'config_id': 'pt193__pt_dominant_failed__sel_rank__r750_j20_g005_m01_b12_p12_pm05', 'max_rank': 750, 'max_jump': 20.0, 'max_positive_jump': 12.0, 'min_gate': 0.05, 'min_model': 0.1, 'min_base_error_diag': 12.0}`.
- PT200: `{'config_id': 'pt193__pt_oracle_shortlist__sel_rank__r100_j8_g001_m01_b5_p5_pm0', 'max_rank': 100, 'max_jump': 8.0, 'max_positive_jump': 5.0, 'min_gate': 0.01, 'min_model': 0.1, 'min_base_error_diag': 5.0, 'oracle_shortlist_le5_diag': True}`.
- TLL196: `{'config_id': 'tll196__tll_oracle_shortlist__sel_rank__new_r1200_j25_s0_b12_pm05', 'sources': 'new', 'max_rank': 1200, 'max_jump': 25.0, 'min_score': 0.0, 'min_base_error_diag': 12.0, 'oracle_shortlist_le5_diag': True}`.
- TLL197: `{'config_id': 'fail196+severe196', 'sources': 'new', 'max_rank': 1200, 'max_jump': 25.0, 'min_score': 0.0, 'min_base_error_diag': 12.0, 'oracle_shortlist_le5_diag': True}`.
- SF204: usa `sf204_profile_count_non_gt > 0` produzido pelo 247.

## Limitacoes de fidelidade

- Os scripts historicos 190/193/196/200 escolhiam configs safe com metricas GT na validacao original.
- Este script nao reescolhe configs; aplica os config_ids finais conhecidos aos candidatos regenerados.
- Quando `SPINAL_248_USE_DIAGNOSTIC_REPLAY_GATES=1`, sao usados `base_abs_error_diag` e oracle shortlist para reproduzir os gates historicos; isso e replay diagnostico, nao inferencia limpa.
- Para inferencia limpa sem GT, definir `SPINAL_248_USE_DIAGNOSTIC_REPLAY_GATES=0`, sabendo que a fidelidade aos gates historicos fica incompleta.

## Diagnostico de leitura

- train2000: {'source': 'train2000', 'centerline_subdir': 'unet_baseline_2000_padding_512_centerline_val_all_3192', 'case_count': 3192, 'mt_rows_read': 1144964, 'mt_rows_passed': 298091, 'pt193_rows_read': 3830400, 'pt193_rows_passed': 11112, 'tll196_rows_read': 3830400, 'tll196_rows_passed': 2418, 'pt200_rows_read': 3830400, 'pt200_rows_passed': 3780, 'tll197_rows_read': 3830400, 'tll197_rows_passed': 2418, 'severe_rows_read': 5357052, 'severe_rows_passed': 58978}.
- trainfull: {'source': 'trainfull', 'centerline_subdir': 'unet_train_full_padding_512_centerline_val_all_3192', 'case_count': 3192, 'mt_rows_read': 1146566, 'mt_rows_passed': 307984, 'pt193_rows_read': 3830400, 'pt193_rows_passed': 11092, 'tll196_rows_read': 3830400, 'tll196_rows_passed': 4086, 'pt200_rows_read': 3830400, 'pt200_rows_passed': 4000, 'tll197_rows_read': 3830400, 'tll197_rows_passed': 4086, 'severe_rows_read': 5368124, 'severe_rows_passed': 58564}.
