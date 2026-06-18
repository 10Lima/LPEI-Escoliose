# Manifest 243

## Objetivo

Regenerar candidatos PT, MT, TL/L e severe-first por CENTERLINE_SUBDIR, sem criar selectors novos.

## Regra de fidelidade

- Regenerar candidatos e permitido.
- Reutilizar candidatos antigos de outra centerline nao e permitido.
- Usar candidatos parciais de diagnosticos antigos como cobertura global nao e permitido.
- GT pode existir em colunas diagnosticas, mas nao pode entrar na selecao/inferencia.

## Fontes verificadas

- train2000: `C:\Users\danib\OneDrive\Ambiente de Trabalho\LPEI\Datasets\Spinal-AI2024-main\Spinal-AI2024-main\processed_padding_512\centerlines\unet_baseline_2000_padding_512_centerline_val_all_3192` (3192 centerlines).
- trainfull: `C:\Users\danib\OneDrive\Ambiente de Trabalho\LPEI\Datasets\Spinal-AI2024-main\Spinal-AI2024-main\processed_padding_512\centerlines\unet_train_full_padding_512_centerline_val_all_3192` (3192 centerlines).

## Auditoria dos regeneradores historicos

- PT (193 -> 201 -> 211/229): blocked. Scripts: `207_gerar_candidatos_pt_severo_checkpoint205_padding_512.py; 229_gerar_candidatos_validar_trainfull_checkpoint205_padding_512.py`. Motivo: O gerador 207 aceita SPINAL_207_CENTERLINE_SUBDIR, mas foi desenhado para PT severo sobre diagnostico206/checkpoint205 e nao cobre obrigatoriamente as 3192 imagens. Usa-lo como PT global violaria a validacao de cobertura.
- MT (190 -> 191): blocked. Scripts: `190_validar_mt188_protegido_checkpoint186_padding_512.py; 125_validar_selector_mt_diversidade_checkpoint114_global_padding_512.py`. Motivo: O 190 consome MT_CANDIDATES_DIR antigo. O 125 e um validador/selector historico que depende de checkpoint93/gate114 e nao expoe um regenerador all-cases por CENTERLINE_SUBDIR.
- TL_L (196 -> 197 -> 198): blocked. Scripts: `159_gerar_candidatos_tll_checkpoint156_relevantes_padding_512.py; 175_gerar_candidatos_tll_remaining_selector173_padding_512.py; 196_validar_tll_focado_checkpoint194_padding_512.py`. Motivo: 159 e 175 aceitam ou reutilizam stack de centerline, mas escolhem casos a partir de diagnosticos historicos parciais. O 196 consome TLL_OLD_CANDIDATES_DIR/TLL159_DIR/TLL175_DIR antigos.
- SEVERE (203 -> 204 -> 205): blocked. Scripts: `203_preparar_severe_first_checkpoint201_padding_512.py; 204_validar_severe_first_pt_tll_checkpoint201_padding_512.py`. Motivo: Severe-first depende dos candidatos PT e TL/L regenerados e do checkpoint201 reaplicado. Sem PT/TL_L all-cases validos, severe_candidates.csv nao pode ser fiel.

## Decisao

- Bloqueado por fidelidade.
- Nenhum ficheiro de candidatos foi marcado como valido.
- O 242 deve continuar bloqueado ate existirem candidatos all-cases regenerados e validados para PT, MT e TL/L.
