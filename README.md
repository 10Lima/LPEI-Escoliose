# LPEI Scoliosis Evaluation Pack

Clean evaluation repository for the LPEI scoliosis project. The project estimates Cobb angles from scoliosis X-rays using spinal segmentation, centerline extraction, geometric Cobb estimation, and a refined checkpoint-based selection pipeline.

This repository is intended for academic assessment and reproducibility of the final evaluation package.

## Overview

The implemented pipeline uses the Spinal-AI2024 dataset and includes:

- preprocessing with padding to `512x512`;
- spinal segmentation with a TensorFlow/Keras U-Net;
- mask post-processing and centerline extraction;
- geometric Cobb angle estimation for PT, MT, and TL/L;
- refined candidate selection, guards, and checkpoint promotion;
- stored reference metrics for the selected final checkpoint.

## Reference Result

The current safe reference checkpoint is:

```text
checkpoint205_checkpoint201_sf204_v1
```

Metrics on the validated 1000-case evaluation window:

| Metric | Value |
|---|---:|
| MAE3 | 3.6247 |
| RMSE3 | 4.7629 |
| <= 5 degrees | 82.4% |
| failures > 5 degrees | 176 |
| severe failures > 8 degrees | 24 |

Reference files are stored in:

```text
results/final/checkpoint205/
```

## Included Data

This evaluation pack includes the Spinal-AI2024 test subset:

```text
raw/images/test/Spinal-AI2024-subset5/
```

It contains 4000 images and the corresponding Cobb ground truth:

```text
processed/cleaned/Cobb_spinal-AI2024-test_gt.txt
processed/cleaned/cobb_test_gt_subset5.csv
```

The full training dataset is not included. See [data/README.md](data/README.md) for the expected full dataset layout.

## Package Contents

| Path | Purpose |
|---|---|
| `raw/images/test/Spinal-AI2024-subset5/` | Included Spinal-AI2024 test images |
| `processed/cleaned/` | Cobb ground-truth files for subset5 |
| `models/` | U-Net checkpoint tracked with Git LFS |
| `results/final/checkpoint205/` | Final reference checkpoint metrics and rows |
| `results/experimental/` | Experimental summaries retained for context |
| `scripts/` | Selected original scripts for pipeline traceability |
| `run_eval.py` | Lightweight smoke-check command |

## Models

The main model is:

```text
models/unet_baseline_2000_padding_512.keras
```

Large `.keras` files are tracked with Git LFS:

```powershell
git lfs install
git lfs pull
```

Fallback: download the model zip from the GitHub Release and extract it into the repository root.

Planned release asset:

```text
lpei-escoliose-models-v1.zip
```

If a `.keras` file opens as text and starts with `version https://git-lfs.github.com/spec/v1`, it is still only a Git LFS pointer.

## Repository Layout

```text
data/         dataset notes and expected full-data layout
models/       model notes and checksums
processed/    cleaned metadata and test ground truth
raw/          included Spinal-AI2024 subset5 images
results/      final and experimental reference summaries
scripts/      selected original pipeline scripts
run_eval.py   smoke-check command wrapper
```

## Setup

Use Python 3.10 or 3.11 if possible.

```powershell
git clone https://github.com/10Lima/LPEI-Escoliose.git
cd LPEI-Escoliose
git lfs pull

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Smoke Test

Run:

```powershell
python run_eval.py smoke --num-images 8
```

This validates:

- the 4000 subset5 images;
- the 4000 Cobb ground-truth rows;
- image-to-ground-truth filename matching;
- the stored checkpoint205 reference metrics.

Expected output includes:

```text
Smoke check OK
Subset5: 4000 images
GT: 4000 rows
Checkpoint205 reference:
  MAE3: 3.6247
```

## Important Notes

- This repository is for academic evaluation and research reproducibility.
- The code and models are not medical devices.
- Predictions must not be used as autonomous clinical diagnoses.
- The `train_full` results included under `results/experimental/` are direct-Cobb experimental summaries and should not be presented as a replacement for the refined checkpoint205 result.

## License and Citation

The source code is released under the MIT License. Included data, model weights, and third-party materials may be subject to their own terms; see [DATA_NOTICE.md](DATA_NOTICE.md).

Citation metadata is provided in [CITATION.cff](CITATION.cff).
