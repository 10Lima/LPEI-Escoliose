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

MAX_TRAIN_SAMPLES = 2000
MAX_VAL_SAMPLES = 400

DATASET_DIR = get_dataset_dir()

# PIPELINE 512
PROCESSED_DIR = DATASET_DIR / "processed_padding_512"

TRAIN_JSON_PATH = PROCESSED_DIR / "splits" / "train.json"
VAL_JSON_PATH = PROCESSED_DIR / "splits" / "val.json"

MODELS_DIR = PROCESSED_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / MODEL_NAME

IMAGE_SIZE = 512

# 512 consome muito mais VRAM
BATCH_SIZE = 2

EPOCHS = 15

LEARNING_RATE = 1e-4


# =========================
# FUNÇÕES
# =========================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


train_data = load_json(TRAIN_JSON_PATH)
val_data = load_json(VAL_JSON_PATH)

train_samples = train_data["samples"][:MAX_TRAIN_SAMPLES]
val_samples = val_data["samples"][:MAX_VAL_SAMPLES]

print("\n===== DATASET 512 =====")
print(f"Train usado: {len(train_samples)}")
print(f"Val usado: {len(val_samples)}")
print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"Batch size: {BATCH_SIZE}")


# =========================
# LOAD IMAGEM + MÁSCARA
# =========================

def load_image_and_mask(image_path, mask_path):

    image = Image.open(image_path).convert("L")
    mask = Image.open(mask_path).convert("L")

    image = np.array(image, dtype=np.float32) / 255.0
    mask = np.array(mask, dtype=np.float32) / 255.0

    mask = (mask > 0.5).astype(np.float32)

    image = np.expand_dims(image, axis=-1)
    mask = np.expand_dims(mask, axis=-1)

    return image, mask


def create_dataset(samples, shuffle=False):

    image_paths = [s["crop_normalized_path"] for s in samples]
    mask_paths = [s["mask_normalized_path"] for s in samples]

    def generator():
        for img_path, mask_path in zip(image_paths, mask_paths):

            image, mask = load_image_and_mask(
                img_path,
                mask_path
            )

            yield image, mask

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(
                shape=(IMAGE_SIZE, IMAGE_SIZE, 1),
                dtype=tf.float32
            ),
            tf.TensorSpec(
                shape=(IMAGE_SIZE, IMAGE_SIZE, 1),
                dtype=tf.float32
            )
        )
    )

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(samples),
            reshuffle_each_iteration=True
        )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


train_dataset = create_dataset(
    train_samples,
    shuffle=True
)

val_dataset = create_dataset(
    val_samples,
    shuffle=False
)


# =========================
# BLOCO U-NET
# =========================

def conv_block(x, filters):

    x = tf.keras.layers.Conv2D(
        filters,
        3,
        padding="same"
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(
        filters,
        3,
        padding="same"
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.ReLU()(x)

    return x


# =========================
# U-NET
# =========================

def build_unet(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)):

    inputs = tf.keras.Input(shape=input_shape)

    # Encoder

    c1 = conv_block(inputs, 32)
    p1 = tf.keras.layers.MaxPooling2D()(c1)

    c2 = conv_block(p1, 64)
    p2 = tf.keras.layers.MaxPooling2D()(c2)

    c3 = conv_block(p2, 128)
    p3 = tf.keras.layers.MaxPooling2D()(c3)

    c4 = conv_block(p3, 256)
    p4 = tf.keras.layers.MaxPooling2D()(c4)

    # Bottleneck

    bn = conv_block(p4, 512)

    # Decoder

    u1 = tf.keras.layers.UpSampling2D()(bn)
    u1 = tf.keras.layers.Concatenate()([u1, c4])
    c5 = conv_block(u1, 256)

    u2 = tf.keras.layers.UpSampling2D()(c5)
    u2 = tf.keras.layers.Concatenate()([u2, c3])
    c6 = conv_block(u2, 128)

    u3 = tf.keras.layers.UpSampling2D()(c6)
    u3 = tf.keras.layers.Concatenate()([u3, c2])
    c7 = conv_block(u3, 64)

    u4 = tf.keras.layers.UpSampling2D()(c7)
    u4 = tf.keras.layers.Concatenate()([u4, c1])
    c8 = conv_block(u4, 32)

    outputs = tf.keras.layers.Conv2D(
        1,
        1,
        activation="sigmoid"
    )(c8)

    model = tf.keras.Model(inputs, outputs)

    return model


# =========================
# MÉTRICAS
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
# BUILD MODELO
# =========================

model = build_unet()

optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE
)

model.compile(
    optimizer=optimizer,
    loss=combined_loss,
    metrics=[
        dice_coefficient,
        "binary_accuracy"
    ]
)

model.summary()


# =========================
# CALLBACKS
# =========================

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_PATH),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
]


# =========================
# TREINO
# =========================

print("\n===== INICIAR TREINO 512 =====")

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=callbacks
)


# =========================
# GRÁFICOS
# =========================

plt.figure(figsize=(10, 5))

plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title("Training Loss - U-Net 512")

plt.legend()

plot_path = MODELS_DIR / "training_loss_512.png"

plt.savefig(plot_path)
plt.close()


# =========================
# FINAL
# =========================

print("\n===== TREINO CONCLUÍDO =====")

print(f"Modelo guardado em: {MODEL_PATH}")
print(f"Gráfico guardado em: {plot_path}")
