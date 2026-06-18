import json
from collections import defaultdict
import statistics
from pathlib import Path

from config import get_dataset_dir

# =========================
# CONFIGURAÇÕES
# =========================


SPLIT = "train"  # mudar para "test" se quiseres analisar o teste
DATASET_DIR = get_dataset_dir()

if SPLIT == "train":
    ANNOTATION_PATH = DATASET_DIR / "Spinal_AI2024_train__annotation" / "Spinal_AI2024_train__annotation.json"
else:
    ANNOTATION_PATH = DATASET_DIR / "Spinal_AI2024_test_annotation" / "Spinal_AI2024_test_annotation.json"

MIN_VERTEBRAE = 14


# =========================
# LER JSON COCO
# =========================

with open(ANNOTATION_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

images = data["images"]
annotations = data["annotations"]


# =========================
# AGRUPAR ANNOTATIONS POR IMAGEM
# =========================

annotations_by_image = defaultdict(list)

for ann in annotations:
    annotations_by_image[ann["image_id"]].append(ann)


# =========================
# ESTATÍSTICAS
# =========================

vertebra_counts = []

for img in images:
    image_id = img["id"]
    vertebra_counts.append(len(annotations_by_image[image_id]))


print("\n===== ANÁLISE GERAL DO DATASET =====")
print(f"Split analisado: {SPLIT}")
print(f"Ficheiro: {ANNOTATION_PATH}")
print(f"Número total de imagens: {len(images)}")
print(f"Número total de annotations/vértebras: {len(annotations)}")

print("\n===== VÉRTEBRAS POR IMAGEM =====")
print(f"Mínimo: {min(vertebra_counts)}")
print(f"Máximo: {max(vertebra_counts)}")
print(f"Média: {statistics.mean(vertebra_counts):.2f}")
print(f"Mediana: {statistics.median(vertebra_counts)}")


# =========================
# POSSÍVEIS PROBLEMAS
# =========================

images_without_annotations = []
images_with_few_vertebrae = []

for img in images:
    image_id = img["id"]
    file_name = img["file_name"]
    count = len(annotations_by_image[image_id])

    if count == 0:
        images_without_annotations.append(file_name)

    if count < MIN_VERTEBRAE:
        images_with_few_vertebrae.append((file_name, count))


print("\n===== POSSÍVEIS PROBLEMAS =====")
print(f"Imagens sem annotations: {len(images_without_annotations)}")
print(f"Imagens com menos de {MIN_VERTEBRAE} vértebras: {len(images_with_few_vertebrae)}")

print("\nExemplos de imagens com poucas vértebras:")
for file_name, count in images_with_few_vertebrae[:10]:
    print(f"{file_name} -> {count} vértebras")


# =========================
# VALIDAR SEGMENTATIONS
# =========================

invalid_segmentations = []

for ann in annotations:
    segmentation = ann.get("segmentation", [])

    if not segmentation:
        invalid_segmentations.append(ann["id"])
        continue

    coords = segmentation[0]

    if len(coords) != 8:
        invalid_segmentations.append(ann["id"])


print("\n===== VALIDAÇÃO DAS SEGMENTATIONS =====")
print(f"Annotations com segmentation inválida: {len(invalid_segmentations)}")


# =========================
# EXEMPLO
# =========================

example_img = images[0]
example_id = example_img["id"]
example_annotations = annotations_by_image[example_id]

print("\n===== EXEMPLO DE UMA IMAGEM =====")
print(f"Imagem: {example_img['file_name']}")
print(f"ID: {example_id}")
print(f"Dimensão: {example_img['width']} x {example_img['height']}")
print(f"Nº de vértebras anotadas: {len(example_annotations)}")

print("\nPrimeiras 3 annotations dessa imagem:")
for ann in example_annotations[:3]:
    print({
        "annotation_id": ann["id"],
        "bbox": ann["bbox"],
        "segmentation": ann["segmentation"]
    })

print("\n===== FIM DA ANÁLISE =====")
