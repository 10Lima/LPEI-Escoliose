# Manifest 242

## Escopo

- Reaplicar a pipeline refinada final mantendo regras/configs fixas.
- Mudar apenas a fonte geometrica: centerline train2000 vs trainfull.
- Nao promover checkpoints.
- Nao criar selectors novos.
- Nao criar guards novos.
- Nao usar GT na inferencia.

## Centerlines verificadas

- train2000: `C:\Users\danib\OneDrive\Ambiente de Trabalho\LPEI\Datasets\Spinal-AI2024-main\Spinal-AI2024-main\processed_padding_512\centerlines\unet_baseline_2000_padding_512_centerline_val_all_3192` (3192).
- trainfull: `C:\Users\danib\OneDrive\Ambiente de Trabalho\LPEI\Datasets\Spinal-AI2024-main\Spinal-AI2024-main\processed_padding_512\centerlines\unet_train_full_padding_512_centerline_val_all_3192` (3192).

## Componentes reutilizaveis diretamente

- Cobb direto base: `232_validar_cobb_trainfull_val_all_3192_padding_512.py`; reutilizavel_com_env. Aceita SPINAL_232_CENTERLINE_SUBDIR.
- PT severo checkpoint205: `207_gerar_candidatos_pt_severo_checkpoint205_padding_512.py`; reutilizavel_com_env. Aceita SPINAL_207_CENTERLINE_SUBDIR; e o precedente usado por 229.
- Diagnostico PT checkpoint205: `206_diagnosticar_pt_severo_sem_oracle_checkpoint205_padding_512.py`; reutilizavel_com_env. Aceita SPINAL_206_CENTERLINE_SUBDIR.
- Promocoes checkpoint: `191, 194, 198, 201, 205`; reutilizavel_se_entradas_equivalentes_existirem. Materializam checkpoints; nao escolhem novas regras.

## Componentes que precisam de refatoracao/regeneracao

- MT 190 -> 191: `190_validar_mt188_protegido_checkpoint186_padding_512.py`. Regenerar MT_CANDIDATES_DIR para cada CENTERLINE_SUBDIR. Bloqueio: Le candidatos antigos de validar_selector_mt_diversidade_checkpoint114_global_top1200_v1.
- PT 193 -> 194 e 200 -> 201: `193_validar_pt_focado_checkpoint191_padding_512.py, 200_validar_pt_residual_checkpoint198_padding_512.py`. Regenerar PT_CANDIDATES_DIR equivalente para cada CENTERLINE_SUBDIR. Bloqueio: 193 le gate_pt_hard_negatives_checkpoint114_v1, calculado sobre artefactos antigos.
- TL/L 196 -> 197 -> 198: `196_validar_tll_focado_checkpoint194_padding_512.py, 197_validar_combo_tll196_checkpoint194_padding_512.py`. Regenerar TLL_OLD_CANDIDATES_DIR, TLL159_DIR e TLL175_DIR para cada CENTERLINE_SUBDIR. Bloqueio: 196 le candidatos antigos e nao troca centerline diretamente neste ponto da cadeia.
- Severe-first 203 -> 204 -> 205: `203_preparar_severe_first_checkpoint201_padding_512.py, 204_validar_severe_first_pt_tll_checkpoint201_padding_512.py`. Regenerar candidatos severe-first PT/TL-L a partir dos candidatos PT/TL-L equivalentes. Bloqueio: 203 depende de PT193/TLL196; sem esses candidatos regenerados por centerline, nao e fiel.
- 211/229: `211_validar_selector_pt_features_v2.py, 229_gerar_candidatos_validar_trainfull_checkpoint205_padding_512.py`. Definir se 211 e parte congelada ou diagnostico OOF antes de incluir na pipeline final. Bloqueio: 211 revalida modelos OOF com labels diagnosticas; nao e, sozinho, um selector congelado.

## Artefactos antigos proibidos nesta comparacao

- `validar_selector_mt_diversidade_checkpoint114_global_top1200_v1`
- `gate_pt_hard_negatives_checkpoint114_v1`
- `gate_two_stage_hard_negatives_tll_v1`
- `gerar_candidatos_tll_checkpoint156_relevantes_v1`
- `gerar_candidatos_tll_remaining_selector173_v1`
- `diagnosticar_falhas_checkpoint191_v1`
- `diagnosticar_falhas_checkpoint194_v1`
- `diagnosticar_falhas_checkpoint198_v1`
- `auditar_gargalo_checkpoint201_v1`
- `diagnostico_pt_severo_sem_oracle_checkpoint205_v1`

## Decisao

- Bloqueado por fidelidade metodologica.
- A comparacao final nao foi calculada porque reutilizar candidatos antigos violaria a regra 'so muda CENTERLINE_SUBDIR'.
