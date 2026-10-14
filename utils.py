"""Utilidades compartidas por train.py, predict.py y app.py.

Todo el preprocesamiento (redimensionado, formato) vive aquí para que el
entrenamiento y la aplicación procesen las imágenes de forma idéntica.
La normalización (÷255) y el data augmentation van DENTRO del modelo.
"""
import random
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

DATASET_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
IMG_SIZE = 128
CLASS_NAMES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]  # orden alfabético = id de clase
NOMBRES_ES = {
    "daisy": "Margarita",
    "dandelion": "Diente de león",
    "roses": "Rosa",
    "sunflowers": "Girasol",
    "tulips": "Tulipán",
}
ARCHIVO_ES = {  # nombres de archivo sin tildes para las imágenes de ejemplo
    "daisy": "margarita",
    "dandelion": "diente_de_leon",
    "roses": "rosa",
    "sunflowers": "girasol",
    "tulips": "tulipan",
}
IMG_EXTS = {".jpg", ".jpeg", ".png"}


# ----------------------------------------------------------------------------
# Datos
# ----------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def download_dataset(data_dir: str = "data") -> Path:
    """Descarga y descomprime Flowers (~218 MB) si no existe. Devuelve la carpeta raíz."""
    data_dir = Path(data_dir)
    root = data_dir / "flower_photos"
    if root.exists() and all((root / c).is_dir() for c in CLASS_NAMES):
        return root

    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "flower_photos.tgz"
    if not archive.exists():
        print("Descargando dataset Flowers (~218 MB)...")
        urllib.request.urlretrieve(DATASET_URL, archive)
    with tarfile.open(archive) as tar:
        try:
            tar.extractall(data_dir, filter="data")
        except TypeError:  # Python < 3.12
            tar.extractall(data_dir)
    return root


def list_images(root):
    """Devuelve (rutas, etiquetas) de todas las imágenes del dataset."""
    paths, labels = [], []
    for i, cls in enumerate(CLASS_NAMES):
        for p in sorted((Path(root) / cls).iterdir()):
            if p.suffix.lower() in IMG_EXTS:
                paths.append(str(p))
                labels.append(i)
    return np.array(paths), np.array(labels, dtype="int32")


def split_data(paths, labels, seed: int = 42):
    """70 % entrenamiento / 15 % validación / 15 % prueba, estratificado."""
    from sklearn.model_selection import train_test_split

    p_tr, p_tmp, y_tr, y_tmp = train_test_split(paths, labels, test_size=0.30, stratify=labels, random_state=seed)
    p_va, p_te, y_va, y_te = train_test_split(p_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=seed)
    return (p_tr, y_tr), (p_va, y_va), (p_te, y_te)


# ----------------------------------------------------------------------------
# Evaluación (textos en español)
# ----------------------------------------------------------------------------
def reporte_clasificacion(y_true, y_pred):
    """Equivalente en español de sklearn.metrics.classification_report, como DataFrame."""
    import pandas as pd
    from sklearn.metrics import classification_report

    nombres = [NOMBRES_ES[c] for c in CLASS_NAMES]
    r = classification_report(
        y_true, y_pred, labels=list(range(len(nombres))), target_names=nombres,
        output_dict=True, zero_division=0,
    )
    filas = {n: r[n] for n in nombres}
    filas["Promedio macro"] = r["macro avg"]
    filas["Promedio ponderado"] = r["weighted avg"]
    df = pd.DataFrame(
        {k: {"precisión": v["precision"], "recall": v["recall"], "f1": v["f1-score"], "soporte": int(v["support"])}
         for k, v in filas.items()}
    ).T
    df.loc["Exactitud (accuracy)"] = [np.nan, np.nan, r["accuracy"], len(y_true)]
    df["soporte"] = df["soporte"].astype(int)
    return df.round(4)


# ----------------------------------------------------------------------------
# Preprocesamiento
# ----------------------------------------------------------------------------
def resize_image(img):
    """img: tensor/array HxWx3 con valores 0-255. Devuelve uint8 de IMG_SIZE x IMG_SIZE x 3."""
    import tensorflow as tf

    img = tf.image.resize(tf.cast(img, tf.float32), (IMG_SIZE, IMG_SIZE))
    return tf.cast(tf.round(tf.clip_by_value(img, 0.0, 255.0)), tf.uint8)


def make_dataset(paths, labels, batch_size: int = 32, shuffle: bool = False, seed: int = 42):
    """Pipeline tf.data: lee, redimensiona, guarda en caché (uint8) y agrupa en lotes float32."""
    import tensorflow as tf

    def _load(path, label):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        return resize_image(img), label

    ds = tf.data.Dataset.from_tensor_slices((np.array(paths, dtype=str), np.array(labels, dtype="int32")))
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE).cache()
    if shuffle:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


# ----------------------------------------------------------------------------
# Modelo
# ----------------------------------------------------------------------------
def build_augmentation():
    """Data augmentation (solo se activa durante el entrenamiento)."""
    from tensorflow import keras
    from tensorflow.keras import layers

    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.15),
            layers.RandomTranslation(0.1, 0.1),
            layers.RandomContrast(0.15),
        ],
        name="data_augmentation",
    )


def build_model(augment: bool = True, dropout: float = 0.4, lr: float = 1e-3):
    """CNN: 4 bloques [Conv-BN-ReLU x2 + MaxPool + Dropout] -> GAP -> Dense -> Softmax.

    Entrada: imágenes 0-255 de IMG_SIZE x IMG_SIZE x 3 (la normalización ÷255 es la
    primera capa del modelo, así la app no necesita normalizar nada).
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="imagen")
    x = inputs
    if augment:
        x = build_augmentation()(x)
    x = layers.Rescaling(1.0 / 255, name="normalizacion")(x)

    for filters in (32, 64, 128, 256):
        for _ in range(2):
            x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.2)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(len(CLASS_NAMES), activation="softmax", name="probabilidades")(x)

    model = keras.Model(inputs, outputs, name="cnn_flores" + ("_aug" if augment else "_sin_aug"))
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
