import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

from config import get_dataset_dir


# =========================
# CONFIGURAÇÕES
# =========================

SPLIT = "train"

DATASET_DIR = get_dataset_dir()

# NOVA PIPELINE 512
PROCESSED_DIR = DATASET_DIR / "processed_padding_512"

INPUT_JSON_PATH = PROCESSED_DIR / f"annotations_clean_{SPLIT}.json"

MASKS_NORMALIZED_DIR = PROCESSED_DIR / "spine_masks_normalized" / SPLIT

SPLITS_DIR = PROCESSED_DIR / "splits"

TRAIN_JSON_PATH = SPLITS_DIR / "train.json"
VAL_JSON_PATH = SPLITS_DIR / "val.json"

VAL_RATIO = 0.2
RANDOM_SEED = 42

IMAGE_SIZE = 512


# =========================
# FUNÇÕES
# =========================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def transform_point_to_normalized(x_original, y_original, roi, transform):

    x_crop = x_original - roi["xmin"]
    y_crop = y_original - roi["ymin"]

    scale = transform["scale"]
    pad_x = transform["pad_x"]
    pad_y = transform["pad_y"]

    x_norm = x_crop * scale + pad_x
    y_norm = y_crop * scale + pad_y

    return x_norm, y_norm


def create_normalized_mask_for_sample(sample):

    roi = sample["spine_roi"]
    transform = sample["normalization_transform"]

    mask = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 0)

    draw = ImageDraw.Draw(mask)

    annotations = sample.get("annotations", [])

    if len(annotations) == 0:
        return None, "sem_annotations"

    valid_polygons = 0

    for ann in annotations:

        segmentation = ann.get("segmentation", [])

        if not segmentation:
            continue

        coords = segmentation[0]

        if len(coords) != 8:
            continue

        polygon = []

        for i in range(0, len(coords), 2):

            x_original = coords[i]
            y_original = coords[i + 1]

            x_norm, y_norm = transform_point_to_normalized(
                x_original,
                y_original,
                roi,
                transform
            )

            polygon.append((x_norm, y_norm))

        draw.polygon(polygon, fill=255)

        valid_polygons += 1

    if valid_polygons == 0:
        return None, "sem_poligonos_validos"

    return mask, None


def validate_pair(crop_path, mask_path):

    crop = Image.open(crop_path).convert("L")
    mask = Image.open(mask_path).convert("L")

    return crop.size == mask.size == (IMAGE_SIZE, IMAGE_SIZE)


# =========================
# SCRIPT PRINCIPAL
# =========================

random.seed(RANDOM_SEED)

MASKS_NORMALIZED_DIR.mkdir(parents=True, exist_ok=True)
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

data = load_json(INPUT_JSON_PATH)

samples = data["samples"]

valid_samples = []
invalid_samples = []

print("\n===== A GERAR MÁSCARAS NORMALIZADAS 512 =====")

for sample in samples:

    file_name = sample["file_name"]

    stem = Path(file_name).stem

    crop_path = Path(sample["crop_normalized_path"])

    mask_name = f"{stem}_mask.png"

    mask_path = MASKS_NORMALIZED_DIR / mask_name

    if not crop_path.exists():

        invalid_samples.append({
            "file_name": file_name,
            "reason": "crop_normalized_nao_encontrado"
        })

        continue

    mask, error = create_normalized_mask_for_sample(sample)

    if error is not None:

        invalid_samples.append({
            "file_name": file_name,
            "reason": error
        })

        continue

    mask.save(mask_path)

    if not validate_pair(crop_path, mask_path):

        invalid_samples.append({
            "file_name": file_name,
            "reason": "crop_mask_tamanhos_diferentes"
        })

        continue

    sample["mask_normalized_path"] = str(mask_path).replace("\\", "/")

    valid_samples.append(sample)


# =========================
# SPLIT TRAIN / VAL
# =========================

random.shuffle(valid_samples)

val_size = int(len(valid_samples) * VAL_RATIO)

val_samples = valid_samples[:val_size]
train_samples = valid_samples[val_size:]


train_output = {
    "dataset": "Spinal-AI2024",
    "split": "train",
    "description": "Samples treino 512x512.",
    "total_samples": len(train_samples),
    "samples": train_samples
}

val_output = {
    "dataset": "Spinal-AI2024",
    "split": "val",
    "description": "Samples validação 512x512.",
    "total_samples": len(val_samples),
    "samples": val_samples
}


save_json(train_output, TRAIN_JSON_PATH)
save_json(val_output, VAL_JSON_PATH)


# =========================
# RESUMO FINAL
# =========================

print("\n===== SCRIPT 3 PADDING 512 CONCLUÍDO =====")

print(f"Total samples lidas: {len(samples)}")
print(f"Máscaras válidas criadas: {len(valid_samples)}")
print(f"Samples inválidas: {len(invalid_samples)}")

print("\n===== SPLIT =====")

print(f"Train: {len(train_samples)}")
print(f"Val: {len(val_samples)}")

print("\n===== OUTPUTS =====")

print(f"Máscaras guardadas em: {MASKS_NORMALIZED_DIR}")
print(f"Train JSON: {TRAIN_JSON_PATH}")
print(f"Val JSON: {VAL_JSON_PATH}")

if invalid_samples:

    print("\n===== PRIMEIROS ERROS =====")

    for item in invalid_samples[:10]:
        print(item)
