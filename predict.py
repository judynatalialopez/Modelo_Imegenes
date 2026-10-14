"""Carga del modelo entrenado y clasificación de imágenes nuevas.

Uso por consola:
    python predict.py ejemplos/rosa_1.jpg ejemplos/tulipan_1.jpg
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from tensorflow import keras

from utils import CLASS_NAMES, NOMBRES_ES, resize_image

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "modelo" / "modelo_flores.keras"


def load_model(path: Path = MODEL_PATH):
    if not Path(path).exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en {path}. Ejecuta primero `python train.py` "
            "o copia aquí la carpeta modelo/ generada en el notebook."
        )
    return keras.models.load_model(path)


def preprocess(image: Image.Image) -> np.ndarray:
    """PIL -> array float32 (1, IMG_SIZE, IMG_SIZE, 3) con valores 0-255."""
    image = ImageOps.exif_transpose(image).convert("RGB")  # corrige la rotación de fotos de celular
    arr = resize_image(np.array(image)).numpy().astype("float32")
    return arr[None, ...]


def predict_images(model, images):
    """Devuelve, por imagen, la clase predicha, la confianza y todas las probabilidades."""
    batch = np.concatenate([preprocess(im) for im in images], axis=0)
    probs = model.predict(batch, verbose=0)
    results = []
    for p in probs:
        i = int(p.argmax())
        results.append({
            "clase": CLASS_NAMES[i],
            "nombre": NOMBRES_ES[CLASS_NAMES[i]],
            "confianza": float(p[i]),
            "probabilidades": {NOMBRES_ES[c]: float(v) for c, v in zip(CLASS_NAMES, p)},
        })
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Uso: python predict.py imagen1.jpg [imagen2.jpg ...]")
    mdl = load_model()
    for path in sys.argv[1:]:
        r = predict_images(mdl, [Image.open(path)])[0]
        print(f"{r['nombre']:15} ({r['confianza']:.1%}) <- {path}")
