"""Aplicación Streamlit: clasificador de flores con una CNN.

Ejecutar:  streamlit run app.py
"""
import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

from predict import load_model, predict_images
from utils import CLASS_NAMES, NOMBRES_ES

BASE_DIR = Path(__file__).resolve().parent
METRICS_PATH = BASE_DIR / "reports" / "metrics.json"
CONFIG_PATH = BASE_DIR / "modelo" / "config.json"
UMBRAL_CONFIANZA = 0.60

st.set_page_config(page_title="Clasificador de flores", page_icon="🌸", layout="centered")

# Streamlit no permite cambiar el idioma de sus widgets; este CSS reemplaza los textos
# en inglés del selector de archivos (si una versión futura cambia el HTML, simplemente no se aplica).
st.markdown(
    """
    <style>
    [data-testid="stFileUploaderDropzoneInstructions"] span,
    [data-testid="stFileUploaderDropzoneInstructions"] small { position: relative; display: block; visibility: hidden; height: 1.3em; }
    [data-testid="stFileUploaderDropzoneInstructions"] span::before,
    [data-testid="stFileUploaderDropzoneInstructions"] small::before { visibility: visible; position: absolute; left: 0; top: 0; white-space: nowrap; }
    [data-testid="stFileUploaderDropzoneInstructions"] span::before { content: "Arrastra y suelta la imagen aquí"; }
    [data-testid="stFileUploaderDropzoneInstructions"] small::before { content: "Límite 200 MB por archivo • JPG, JPEG, PNG"; }
    [data-testid="stFileUploaderDropzone"] button { font-size: 0; }
    [data-testid="stFileUploaderDropzone"] button::after { content: "Buscar archivos"; font-size: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Cargando modelo...")
def get_model():
    return load_model()


def read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


# ----------------------------------------------------------------------------
# Barra lateral
# ----------------------------------------------------------------------------
config, metrics = read_json(CONFIG_PATH), read_json(METRICS_PATH)
with st.sidebar:
    st.header("Sobre el modelo")
    st.write(
        "Red neuronal convolucional (CNN) entrenada desde cero con el dataset **Flowers** "
        "(~3.600 fotos). Reconoce 5 tipos de flor:"
    )
    st.markdown("\n".join(f"- {NOMBRES_ES[c]}" for c in CLASS_NAMES))
    if config:
        st.caption(f"Entrada: {config['img_size']}×{config['img_size']} px · "
                   f"Data augmentation: {'sí' if config['augmentation'] else 'no'}")
    if metrics:
        st.subheader("Métricas en test")
        c1, c2 = st.columns(2)
        c1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        c2.metric("F1 macro", f"{metrics['f1_macro']:.3f}")
        c1.metric("Precisión macro", f"{metrics['precision_macro']:.3f}")
        c2.metric("Recall macro", f"{metrics['recall_macro']:.3f}")

# ----------------------------------------------------------------------------
# Cuerpo principal
# ----------------------------------------------------------------------------
st.title("🌸 Clasificador de flores")
st.write("Sube una foto de una **margarita, diente de león, rosa, girasol o tulipán** y pulsa **Clasificar**.")

archivo = st.file_uploader("Imagen", type=["jpg", "jpeg", "png"])

if archivo is None:
    st.info("Sube una imagen para comenzar. Puedes usar las de la carpeta `ejemplos/`.")
else:
    try:
        imagen = ImageOps.exif_transpose(Image.open(archivo)).convert("RGB")
    except Exception:
        st.error("No se pudo leer la imagen. Prueba con otro archivo JPG o PNG.")
        st.stop()

    st.image(imagen, caption=archivo.name, use_container_width=True)

    if st.button("Clasificar", type="primary"):
        try:
            r = predict_images(get_model(), [imagen])[0]
        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()

        st.subheader(f"🌼 {r['nombre']} — confianza {r['confianza']:.1%}")
        st.progress(min(max(r["confianza"], 0.0), 1.0))

        if r["confianza"] < UMBRAL_CONFIANZA:
            st.warning(
                "Confianza baja. Puede que la imagen no sea una de las 5 flores que conoce el modelo "
                "o que la foto esté poco clara. El modelo siempre elige una de sus 5 clases."
            )

        df = pd.DataFrame({"clase": list(r["probabilidades"]), "probabilidad": list(r["probabilidades"].values())})
        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                y=alt.Y("clase:N", sort="-x", title=None),
                x=alt.X("probabilidad:Q", scale=alt.Scale(domain=[0, 1]), title="Probabilidad"),
                tooltip=["clase", alt.Tooltip("probabilidad:Q", format=".1%")],
            )
        )
        st.altair_chart(chart, use_container_width=True)
