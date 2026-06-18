# Manifest 244

## Objetivo

Gerar candidatos PT all-cases por CENTERLINE_SUBDIR usando a logica do 207.

## Fidelidade

- Bandas PT iguais ao 207.
- Smooth values iguais ao 207 por default.
- Exportacao diversa: top_score, low_cobb, high_cobb, low_jump.
- GT usado apenas para diagnostico oracle.
- Nao escolhe candidato final.
- Nao cria selector nem guard.

## Configuracao

- SMOOTH_VALUES: `[0.0, 30.0, 60.0, 120.0]`.
- MIN_POINT_DISTANCES: `[15.0, 35.0]`.
- MIN_NORM_SPANS: `[0.04, 0.12]`.
- MAX_EXPORT_PER_COMBO: `50`.
- MAX_DIVERSE_PER_COMBO: `25`.
- MAX_EXPORT_PER_FILE: `1200`.
- MAX_FILES: `0`.

## Outputs

- train2000: `/content/LPEI/processed_padding_512/cobb_results/comparacao_centerline_train2000_vs_trainfull_pipeline_refinada_final_v1/train2000/candidates_valid/pt_candidates.csv`.
- trainfull: `/content/LPEI/processed_padding_512/cobb_results/comparacao_centerline_train2000_vs_trainfull_pipeline_refinada_final_v1/trainfull/candidates_valid/pt_candidates.csv`.
