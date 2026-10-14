"""Entrenamiento de una CNN para clasificar flores (5 clases).

Uso:
    python train.py                       # con data augmentation (recomendado)
    python train.py --no-augmentation     # para comparar
    python train.py --epochs 60 --batch-size 64

Genera:
    modelo/modelo_flores.keras      -> modelo entrenado (incluye normalización)
    modelo/config.json              -> hiperparámetros y clases
    reports/metrics.json            -> métricas en test
    reports/*.png                   -> curvas, matriz de confusión, predicciones
    reports/reporte_clasificacion.csv -> precisión, recall y F1 por clase
    reports/predicciones_test.csv   -> 12 predicciones de ejemplo
    ejemplos/                       -> 10 imágenes de test para probar la app
"""
import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tensorflow import keras

from utils import (
    CLASS_NAMES,
    IMG_SIZE,
    ARCHIVO_ES,
    NOMBRES_ES,
    build_model,
    download_dataset,
    list_images,
    make_dataset,
    reporte_clasificacion,
    set_seed,
    split_data,
)

NOMBRES = [NOMBRES_ES[c] for c in CLASS_NAMES]


def plot_history(history, path):
    h = history.history
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(h["loss"], label="entrenamiento")
    axes[0].plot(h["val_loss"], label="validación")
    axes[0].set(title="Pérdida", xlabel="Época", ylabel="Crossentropy")
    axes[0].legend()
    axes[1].plot(h["accuracy"], label="entrenamiento")
    axes[1].plot(h["val_accuracy"], label="validación")
    axes[1].set(title="Exactitud", xlabel="Época", ylabel="Accuracy")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def evaluate(model, test_ds, y_test, reports_dir: Path):
    probs = model.predict(test_ds, verbose=0)
    preds = probs.argmax(axis=1)

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision_macro": precision_score(y_test, preds, average="macro"),
        "recall_macro": recall_score(y_test, preds, average="macro"),
        "f1_macro": f1_score(y_test, preds, average="macro"),
        "f1_weighted": f1_score(y_test, preds, average="weighted"),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        "n_test": int(len(y_test)),
    }
    print("\n=== Evaluación en TEST ===")
    reporte = reporte_clasificacion(y_test, preds)
    print(reporte.to_string())
    reporte.to_csv(reports_dir / "reporte_clasificacion.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, preds, display_labels=NOMBRES, cmap="Blues", ax=axes[0], colorbar=False, xticks_rotation=40
    )
    axes[0].set_title("Matriz de confusión (conteos)")
    ConfusionMatrixDisplay.from_predictions(
        y_test, preds, display_labels=NOMBRES, cmap="Blues", ax=axes[1], colorbar=False,
        normalize="true", values_format=".2f", xticks_rotation=40,
    )
    axes[1].set_title("Matriz de confusión (normalizada por clase real)")
    for ax in axes:
        ax.set(xlabel="Clase predicha", ylabel="Clase real")
    plt.tight_layout()
    plt.savefig(reports_dir / "matriz_confusion.png", dpi=130)
    plt.close()
    return metrics, probs, preds


def plot_predictions(paths, y_true, probs, reports_dir: Path, n=12, seed=42):
    """Cuadrícula con n predicciones (imagen + real + predicha + confianza) y CSV."""
    idx = np.random.RandomState(seed).choice(len(paths), size=min(n, len(paths)), replace=False)
    cols = 4
    rows = int(np.ceil(len(idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.6 * rows))
    records = []
    for ax, i in zip(np.array(axes).ravel(), idx):
        pred = int(probs[i].argmax())
        ok = pred == int(y_true[i])
        ax.imshow(Image.open(paths[i]).convert("RGB"))
        ax.set_title(
            f"Real: {NOMBRES[y_true[i]]}\nPred: {NOMBRES[pred]} ({probs[i][pred]:.0%})",
            fontsize=9, color="green" if ok else "red",
        )
        ax.axis("off")
        records.append({"archivo": Path(paths[i]).name, "real": NOMBRES[y_true[i]], "prediccion": NOMBRES[pred],
                        "confianza": round(float(probs[i][pred]), 4), "correcta": ok})
    for ax in np.array(axes).ravel()[len(idx):]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(reports_dir / "predicciones_test.png", dpi=130)
    plt.close()
    df = pd.DataFrame(records)
    df.to_csv(reports_dir / "predicciones_test.csv", index=False)
    print("\n=== Predicciones de ejemplo (test) ===")
    print(df.to_string(index=False))


def copy_examples(paths, y_test, examples_dir: Path, per_class=2):
    """Copia imágenes de test a ejemplos/ para probarlas en la app."""
    examples_dir.mkdir(parents=True, exist_ok=True)
    for i, cls in enumerate(CLASS_NAMES):
        for n, p in enumerate(paths[y_test == i][:per_class], start=1):
            shutil.copy(p, examples_dir / f"{ARCHIVO_ES[cls]}_{n}{Path(p).suffix.lower()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-augmentation", action="store_true", help="Desactiva el data augmentation")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="modelo")
    ap.add_argument("--reports-dir", default="reports")
    ap.add_argument("--examples-dir", default="ejemplos")
    args = ap.parse_args()

    out_dir, reports_dir = Path(args.out_dir), Path(args.reports_dir)
    out_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)
    set_seed(args.seed)

    root = download_dataset(args.data_dir)
    paths, labels = list_images(root)
    print("Imágenes por clase:", dict(zip(NOMBRES, np.bincount(labels))))

    (p_tr, y_tr), (p_va, y_va), (p_te, y_te) = split_data(paths, labels, args.seed)
    print(f"Train: {len(p_tr)} | Val: {len(p_va)} | Test: {len(p_te)}")

    train_ds = make_dataset(p_tr, y_tr, args.batch_size, shuffle=True, seed=args.seed)
    val_ds = make_dataset(p_va, y_va, args.batch_size)
    test_ds = make_dataset(p_te, y_te, args.batch_size)

    model = build_model(augment=not args.no_augmentation, dropout=args.dropout, lr=args.lr)
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
        keras.callbacks.ModelCheckpoint(out_dir / "checkpoint.keras", monitor="val_loss", save_best_only=True),
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks, verbose=2)
    plot_history(history, reports_dir / "curvas_entrenamiento.png")

    metrics, probs, _ = evaluate(model, test_ds, y_te, reports_dir)
    plot_predictions(p_te, y_te, probs, reports_dir, seed=args.seed)
    copy_examples(p_te, y_te, Path(args.examples_dir))

    model_path = out_dir / "modelo_flores.keras"
    model.save(model_path)
    config = {**vars(args), "img_size": IMG_SIZE, "class_names": CLASS_NAMES, "nombres_es": NOMBRES,
              "augmentation": not args.no_augmentation, "epochs_entrenadas": len(history.history["loss"])}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))
    (reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nModelo guardado en: {model_path}")


if __name__ == "__main__":
    main()
