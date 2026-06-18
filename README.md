# LPEI-Escoliose

Pacote local de staging para a versao limpa do projeto de Laboratorio de Projeto em Engenharia Informatica sobre analise automatica de escoliose em radiografias.

Este staging ainda nao e o repositorio final. Foi criado para selecionar e validar os ficheiros antes de qualquer commit ou push.

## Objetivo

O projeto estima automaticamente os angulos de Cobb PT, MT e TL/L a partir de radiografias do dataset Spinal-AI2024, usando:

- preprocessamento com padding para 512x512;
- segmentacao da coluna com U-Net em TensorFlow/Keras;
- extracao de centerline;
- calculo geometrico dos angulos de Cobb;
- pipeline refinada com candidatos, guards e checkpoints.

## Resultado de referencia

O checkpoint seguro atual e `checkpoint205_checkpoint201_sf204_v1`.

Metricas em 1000 casos:

- MAE3: `3.6247`
- RMSE3: `4.7629`
- erro <= 5 graus: `82.4%`
- falhas > 5 graus: `176`
- falhas severas > 8 graus: `24`

Os ficheiros deste resultado estao em:

```text
results/final/checkpoint205/
```

## Modelos

O checkpoint principal esperado e:

```text
models/unet_baseline_2000_padding_512.keras
```

No repositorio final, ficheiros `.keras` devem ser obtidos por Git LFS:

```powershell
git lfs install
git lfs pull
```

Fallback recomendado: descarregar o zip de modelos da GitHub Release e extrair na raiz do repositorio.

Nome previsto do asset:

```text
lpei-escoliose-models-v1.zip
```

Se um ficheiro `.keras` abrir como texto e comecar por `version https://git-lfs.github.com/spec/v1`, entao ainda e apenas um pointer LFS.

## Estrutura

```text
data/                 instrucoes para obter/preparar dados
raw/                  subset de avaliacao, se for autorizado incluir
models/               instrucoes para obter checkpoints
scripts/              scripts selecionados, ainda com nomes originais
results/              resultados finais pequenos e summaries experimentais
assets/               figuras futuras
archive/              historico pequeno, se necessario
```

## Smoke test

Este staging inclui um smoke check inicial:

```powershell
python run_eval.py smoke --num-images 8
```

Este comando valida:

- existencia das 4000 imagens do subset5;
- existencia das 4000 linhas de ground truth Cobb;
- correspondencia entre nomes de imagem e GT;
- presenca das metricas finais do checkpoint205.

## Estado dos dados

Nesta versao de staging, o dataset `subset5` ja foi copiado para:

```text
raw/images/test/Spinal-AI2024-subset5/
```

Tambem foi copiado o ground truth de teste:

```text
processed/cleaned/Cobb_spinal-AI2024-test_gt.txt
processed/cleaned/cobb_test_gt_subset5.csv
```

O checkpoint principal `.keras` foi copiado para `models/`.

Decisao pendente:

- publicar modelos via GitHub Release e/ou Git LFS no repositorio final.

## Aviso

Este projeto e academico e nao e um dispositivo medico. As estimativas nao devem ser usadas como diagnostico clinico autonomo.
