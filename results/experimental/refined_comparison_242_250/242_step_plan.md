# Plano 242 - reaplicar pipeline refinada final

## Objetivo

Comparar o efeito da centerline mantendo a pipeline Cobb fixa.

## Fontes

- train2000: `unet_baseline_2000_padding_512_centerline_val_all_3192` (3192 centerlines).
- trainfull: `unet_train_full_padding_512_centerline_val_all_3192` (3192 centerlines).

## Cadeia historica alvo

- MT: 190 -> 191.
- PT: 193 -> 194, 200 -> 201, e 211/229 apenas se for confirmado como parte congelada.
- TL/L: 196 -> 197 -> 198.
- Severe-first: 203 -> 204 -> 205.

## Passos fieis necessarios

1. Calcular Cobb direto base para cada centerline.
2. Materializar checkpoint inicial compativel para cada fonte.
3. Regenerar candidatos MT/PT/TL-L para cada fonte, com os mesmos geradores/configs historicos.
4. Reaplicar 190, 193, 196/197, 200, 203/204 na mesma ordem.
5. Materializar 191, 194, 198, 201 e 205.
6. Comparar train2000 vs trainfull apenas no checkpoint final reaplicado.

## Estado atual

Bloqueado antes da execucao: nem todos os geradores historicos de candidatos estao expostos por CENTERLINE_SUBDIR.
