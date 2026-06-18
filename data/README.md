# Dados

Este staging inclui o subset de teste `Spinal-AI2024-subset5`.

## Opcao recomendada para avaliacao

Subset incluido:

```text
raw/images/test/Spinal-AI2024-subset5/
```

No projeto local original, este subset tem:

- 4000 imagens;
- cerca de 86.91 MB.

Ground truth incluido:

```text
processed/cleaned/Cobb_spinal-AI2024-test_gt.txt
processed/cleaned/cobb_test_gt_subset5.csv
```

## Dataset completo

Para reproduzir treino e preprocessamento completo, e necessaria a estrutura original:

```text
Spinal-AI2024-subset1/
Spinal-AI2024-subset2/
Spinal-AI2024-subset3/
Spinal-AI2024-subset4/
Spinal-AI2024-subset5/
Spinal_AI2024_train__annotation/
Spinal_AI2024_test_annotation/
Cobb_spinal-AI2024-train_gt.txt
Cobb_spinal-AI2024-test_gt.txt
```

Por defeito, os scripts procuram estes dados na raiz definida por `SPINAL_AI_DATASET_DIR`.
