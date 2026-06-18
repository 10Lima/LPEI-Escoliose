import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

from config import get_dataset_dir


# =========================
# CONFIGURAÇÕES
# =========================

MODEL_NAME = "unet_baseline_2000_padding_512.keras"

OUTPUT_SUBDIR = "unet_baseline_2000_padding_512_prob"

DATASET_DIR = get_dataset_dir()

PROCESSED_DIR = DATASET_DIR / "processed_padding_512"

VAL_JSON_PATH = PROCESSED_DIR / "splits" / "val.json"

MODEL_PATH = PROCESSED_DIR / "models" / MODEL_NAME

OUTPUT_DIR = PROCESSED_DIR / "predictions" / OUTPUT_SUBDIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 512
NUM_EXAMPLES = 10
THRESHOLD = 0.5


# =========================
# MÉTRICAS / LOSS
# =========================

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(y_true_f * y_pred_f)

    return (
        (2.0 * intersection + smooth)
        /
        (
            tf.reduce_sum(y_true_f)
            + tf.reduce_sum(y_pred_f)
            + smooth
        )
    )


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)


def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(
        y_true,
        y_pred
    )

    dloss = dice_loss(y_true, y_pred)

    return bce + dloss


# =========================
# FUNÇÕES
# =========================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_image_and_mask(image_path, mask_path):
    image = Image.open(image_path).convert("L")
    mask = Image.open(mask_path).convert("L")

    image_array = np.array(image, dtype=np.float32) / 255.0
    mask_array = np.array(mask, dtype=np.float32) / 255.0

    mask_array = (mask_array > 0.5).astype(np.float32)

    input_image = np.expand_dims(image_array, axis=-1)
    input_image = np.expand_dims(input_image, axis=0)

    return image_array, mask_array, input_image


# =========================
# CARREGAR MODELO
# =========================

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "dice_coefficient": dice_coefficient,
        "dice_loss": dice_loss,
        "combined_loss": combined_loss
    }
)

print("\n===== MODELO 512 CARREGADO =====")
print(MODEL_PATH)


# =========================
# LER VALIDAÇÃO
# =========================

val_data = load_json(VAL_JSON_PATH)
val_samples = val_data["samples"]

print(f"\nTotal validation samples: {len(val_samples)}")


# =========================
# PREVISÕES
# =========================

for i, sample in enumerate(val_samples[:NUM_EXAMPLES]):

    image_path = sample["crop_normalized_path"]
    mask_path = sample["mask_normalized_path"]

    image_array, mask_array, input_image = load_image_and_mask(
        image_path,
        mask_path
    )

    prediction = model.predict(
        input_image,
        verbose=0
    )[0, :, :, 0]

    prediction_binary = (
        prediction > THRESHOLD
    ).astype(np.float32)

    file_name = Path(sample["file_name"]).stem

    print(
        f"{file_name} | "
        f"min: {prediction.min():.4f} | "
        f"max: {prediction.max():.4f} | "
        f"mean: {prediction.mean():.4f}"
    )

    output_path = OUTPUT_DIR / f"{i+1:02d}_{file_name}_comparison_512.png"

    plt.figure(figsize=(16, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(image_array, cmap="gray")
    plt.title("Imagem 512")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(mask_array, cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(
        prediction,
        cmap="gray",
        vmin=0,
        vmax=1
    )
    plt.title("Prediction prob.")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(
        prediction_binary,
        cmap="gray"
    )
    plt.title("Prediction binária")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Guardado: {output_path}")


print("\n===== PREVISÕES 512 CONCLUÍDAS =====")
print(f"Resultados guardados em: {OUTPUT_DIR}")
