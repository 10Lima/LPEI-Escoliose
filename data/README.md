# Data

This evaluation pack includes the Spinal-AI2024 test subset:

```text
raw/images/test/Spinal-AI2024-subset5/
```

Local package contents:

- 4000 test images;
- approximately 86.91 MB;
- Cobb ground truth in `processed/cleaned/`.

Included ground-truth files:

```text
processed/cleaned/Cobb_spinal-AI2024-test_gt.txt
processed/cleaned/cobb_test_gt_subset5.csv
```

## Full Dataset Layout

To reproduce preprocessing and training from scratch, the full original dataset is expected in this layout:

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

By default, the original scripts resolve the dataset root from `SPINAL_AI_DATASET_DIR`. If it is not set, they use the repository root.

