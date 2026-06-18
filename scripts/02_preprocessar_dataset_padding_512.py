import json
from collections import defaultdict
from pathlib import Path

from PIL import Image

from config import get_dataset_dir


# =========================
# CONFIGURAÇÕES
# =========================

SPLIT = "train"

DATASET_DIR = get_dataset_dir()

ANNOTATION_PATH = DATASET_DIR / "Spinal_AI2024_train__annotation" / "Spinal_AI2024_train__annotation.json"
COBB_TXT_PATH = DATASET_DIR / "Cobb_spinal-AI2024-train_gt.txt"

SUBSET_DIRS = [
    DATASET_DIR / "Spinal-AI2024-subset1",
    DATASET_DIR / "Spinal-AI2024-subset2",
    DATASET_DIR / "Spinal-AI2024-subset3",
    DATASET_DIR / "Spinal-AI2024-subset4",
    DATASET_DIR / "Spinal-AI2024-subset5",
]

# Nova pasta para o pipeline 512
PROCESSED_DIR = DATASET_DIR / "processed_padding_512"

CROPS_ORIGINAL_DIR = PROCESSED_DIR / "crops_original" / SPLIT
CROPS_NORMALIZED_DIR = PROCESSED_DIR / "crops_normalized" / SPLIT

OUTPUT_JSON_PATH = PROCESSED_DIR / f"annotations_clean_{SPLIT}.json"

MIN_VERTEBRAE = 14
PADDING_ROI = 20
IMAGE_SIZE = 512


# =========================
# FUNÇÕES
# =========================

def load_cobb_txt(path):
    cobb = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split(",")

            if len(parts) != 4:
                continue

            file_name, pt, mt, tl_l = parts

            cobb[file_name] = {
                "PT": float(pt),
                "MT": float(mt),
                "TL_L": float(tl_l)
            }

    return cobb


def find_image_path(file_name):
    for subset_dir in SUBSET_DIRS:
        possible_path = subset_dir / file_name

        if possible_path.exists():
            return possible_path

    return None


def get_points(annotations):
    points = []

    for ann in annotations:
        segmentation = ann.get("segmentation", [])

        if not segmentation:
            continue

        coords = segmentation[0]

        if len(coords) != 8:
            continue

        for i in range(0, len(coords), 2):
            points.append((coords[i], coords[i + 1]))

    return points


def calculate_roi(points, width, height, padding):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    xmin = max(0, int(min(xs) - padding))
    ymin = max(0, int(min(ys) - padding))
    xmax = min(width, int(max(xs) + padding))
    ymax = min(height, int(max(ys) + padding))

    return {
        "xmin": xmin,
        "ymin": ymin,
        "xmax": xmax,
        "ymax": ymax,
        "padding": padding
    }


def calculate_centroids(annotations):
    centroids = []

    for ann in annotations:
        segmentation = ann.get("segmentation", [])

        if not segmentation:
            continue

        coords = segmentation[0]

        if len(coords) != 8:
            continue

        xs = coords[0::2]
        ys = coords[1::2]

        centroids.append({
            "annotation_id": ann["id"],
            "x": round(sum(xs) / len(xs), 2),
            "y": round(sum(ys) / len(ys), 2)
        })

    return sorted(centroids, key=lambda p: p["y"])


def resize_with_padding(image, target_size=512):
    original_width, original_height = image.size

    scale = min(
        target_size / original_width,
        target_size / original_height
    )

    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    resized = image.resize(
        (new_width, new_height),
        resample=Image.BILINEAR
    )

    padded = Image.new("L", (target_size, target_size), 0)

    paste_x = (target_size - new_width) // 2
    paste_y = (target_size - new_height) // 2

    padded.paste(resized, (paste_x, paste_y))

    transform = {
        "original_crop_width": original_width,
        "original_crop_height": original_height,
        "normalized_width": new_width,
        "normalized_height": new_height,
        "target_size": target_size,
        "scale": scale,
        "pad_x": paste_x,
        "pad_y": paste_y
    }

    return padded, transform


def save_crops(image_path, original_crop_path, normalized_crop_path, roi):
    image = Image.open(image_path).convert("L")

    crop = image.crop((
        roi["xmin"],
        roi["ymin"],
        roi["xmax"],
        roi["ymax"]
    ))

    normalized_crop, transform = resize_with_padding(
        crop,
        target_size=IMAGE_SIZE
    )

    original_crop_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_crop_path.parent.mkdir(parents=True, exist_ok=True)

    crop.save(original_crop_path)
    normalized_crop.save(normalized_crop_path)

    return transform


# =========================
# SCRIPT PRINCIPAL
# =========================

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
CROPS_ORIGINAL_DIR.mkdir(parents=True, exist_ok=True)
CROPS_NORMALIZED_DIR.mkdir(parents=True, exist_ok=True)

with open(ANNOTATION_PATH, "r", encoding="utf-8") as f:
    coco = json.load(f)

images = coco["images"]
annotations = coco["annotations"]

cobb_data = load_cobb_txt(COBB_TXT_PATH)

annotations_by_image = defaultdict(list)

for ann in annotations:
    annotations_by_image[ann["image_id"]].append(ann)

clean_samples = []
removed_samples = []

for img in images:
    image_id = img["id"]
    file_name = img["file_name"]
    width = img["width"]
    height = img["height"]

    img_annotations = annotations_by_image[image_id]
    vertebra_count = len(img_annotations)

    if vertebra_count < MIN_VERTEBRAE:
        removed_samples.append({
            "image_id": image_id,
            "file_name": file_name,
            "reason": f"menos_de_{MIN_VERTEBRAE}_vertebras",
            "vertebra_count": vertebra_count
        })
        continue

    if file_name not in cobb_data:
        removed_samples.append({
            "image_id": image_id,
            "file_name": file_name,
            "reason": "sem_cobb_target",
            "vertebra_count": vertebra_count
        })
        continue

    points = get_points(img_annotations)

    if len(points) == 0:
        removed_samples.append({
            "image_id": image_id,
            "file_name": file_name,
            "reason": "sem_pontos_validos",
            "vertebra_count": vertebra_count
        })
        continue

    image_path = find_image_path(file_name)

    if image_path is None:
        removed_samples.append({
            "image_id": image_id,
            "file_name": file_name,
            "reason": "imagem_nao_encontrada"
        })
        continue

    roi = calculate_roi(points, width, height, PADDING_ROI)

    original_crop_path = CROPS_ORIGINAL_DIR / file_name
    normalized_crop_path = CROPS_NORMALIZED_DIR / file_name

    normalization_transform = save_crops(
        image_path,
        original_crop_path,
        normalized_crop_path,
        roi
    )

    sample = {
        "split": SPLIT,
        "image_id": image_id,
        "file_name": file_name,
        "image_path": str(image_path).replace("\\", "/"),

        "crop_original_path": str(original_crop_path).replace("\\", "/"),
        "crop_normalized_path": str(normalized_crop_path).replace("\\", "/"),

        "width": width,
        "height": height,
        "vertebra_count": vertebra_count,

        "annotations": img_annotations,
        "spine_roi": roi,
        "normalization_transform": normalization_transform,

        "centroids": calculate_centroids(img_annotations),
        "cobb_angles": cobb_data[file_name],

        "future_paths": {
            "spine_mask_normalized_path": str(
                PROCESSED_DIR / "spine_masks_normalized" / SPLIT / file_name.replace(".jpg", "_mask.png")
            ).replace("\\", "/"),
            "centerline_path": str(
                PROCESSED_DIR / "centerlines" / SPLIT / file_name.replace(".jpg", "_centerline.png")
            ).replace("\\", "/")
        }
    }

    clean_samples.append(sample)


output = {
    "dataset": "Spinal-AI2024",
    "split": SPLIT,
    "description": "Dataset 512x512 com crop ROI original e crop normalizado por resize proporcional + padding.",
    "total_images_original": len(images),
    "total_samples_clean": len(clean_samples),
    "total_samples_removed": len(removed_samples),
    "cleaning_criteria": {
        "minimum_vertebrae": MIN_VERTEBRAE,
        "roi_padding": PADDING_ROI,
        "normalized_image_size": IMAGE_SIZE,
        "requires_cobb_target": True,
        "requires_valid_segmentation": True,
        "requires_image_file": True
    },
    "samples": clean_samples,
    "removed_samples": removed_samples
}

with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=4, ensure_ascii=False)


print("\n===== SCRIPT 2 PADDING 512 CONCLUÍDO =====")
print(f"Split: {SPLIT}")
print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"Total original: {len(images)}")
print(f"Total clean: {len(clean_samples)}")
print(f"Total removidas: {len(removed_samples)}")
print(f"JSON criado: {OUTPUT_JSON_PATH}")
print(f"Crops originais guardados em: {CROPS_ORIGINAL_DIR}")
print(f"Crops normalizados 512 guardados em: {CROPS_NORMALIZED_DIR}")
