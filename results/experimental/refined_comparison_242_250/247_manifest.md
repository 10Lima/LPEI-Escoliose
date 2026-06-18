# Manifest 247

## Objetivo

Regenerar a frente severe-first por CENTERLINE_SUBDIR usando os candidatos PT e TL/L ja regenerados.

## Fidelidade

- Usa os perfis PT/TL-L do 204 como anotacao de shortlist.
- Nao cria selector novo.
- Nao cria guard novo.
- Nao escolhe checkpoint final.
- Nao usa GT para exportar candidatos severe-first.
- O gate historico `min_base_error` do 204 depende de erro contra GT; por isso fica apenas como coluna diagnostica.
- O `model_score` PT historico nao existe nos candidatos 244; quando ausente, o script nao o usa para excluir candidatos e assinala `model_score_missing_for_pt204=1`.

## Configuracao

- MAX_FILES: `0`.

## Outputs

- train2000: `C:\Users\danib\OneDrive\Ambiente de Trabalho\LPEI\Datasets\Spinal-AI2024-main\Spinal-AI2024-main\processed_padding_512\cobb_results\comparacao_centerline_train2000_vs_trainfull_pipeline_refinada_final_v1\train2000\candidates_valid\severe_candidates.csv`.
- trainfull: `C:\Users\danib\OneDrive\Ambiente de Trabalho\LPEI\Datasets\Spinal-AI2024-main\Spinal-AI2024-main\processed_padding_512\cobb_results\comparacao_centerline_train2000_vs_trainfull_pipeline_refinada_final_v1\trainfull\candidates_valid\severe_candidates.csv`.
