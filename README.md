# 🌸 Clasificación de flores con una CNN

Proyecto de Deep Learning que clasifica fotos de flores en 5 clases (**margarita, diente de león,
rosa, girasol y tulipán**) con una Red Neuronal Convolucional entrenada desde cero en
TensorFlow/Keras, más una aplicación en Streamlit para subir una imagen y ver la predicción con su confianza.

## 1. Dataset

- **Flowers (`flower_photos`)**, publicado por el equipo de TensorFlow: ~3.670 fotos reales en 5 clases.
- Las imágenes tienen tamaños y proporciones distintos, con fondos y luz variables.
- Se descarga automáticamente al entrenar (~218 MB) en `data/` y **no** se sube a Git.
- Fuente: https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz

## 2. Estructura del repositorio

```
proyecto-cnn/
├── notebook_experimentacion.ipynb   # Google Colab: exploración, experimentos, evaluación
├── train.py                         # Entrenamiento por línea de comandos
├── predict.py                       # Carga del modelo y predicción de imágenes
├── app.py                           # Aplicación Streamlit
├── utils.py                         # Datos, preprocesamiento y arquitectura (compartido)
├── modelo/                          # modelo_flores.keras + config.json
├── reports/                         # gráficas, métricas y predicciones de test
├── ejemplos/                        # imágenes de test para probar la app
├── data/                            # dataset (se descarga solo)
├── requirements.txt
└── README.md
```

## 3. Metodología

1. **Exploración**: balance de clases, cuadrícula de imágenes por clase, distribución de tamaños y proporciones.
2. **Preprocesamiento**: redimensionado a 128×128. La **normalización (÷255)** es la primera capa del modelo,
   así el modelo guardado recibe píxeles crudos y la app no puede normalizar mal.
3. **División**: 70 % entrenamiento / 15 % validación / 15 % prueba, estratificada y con semilla fija.
4. **Data Augmentation** (solo en entrenamiento): volteo horizontal, rotación, zoom, desplazamiento y contraste.
   Se compara **con vs. sin augmentation** y se elige por accuracy de validación.
5. **CNN**: 4 bloques `[Conv-BN-ReLU ×2 → MaxPool → Dropout]` (32/64/128/256 filtros) → GlobalAveragePooling → Dense(128) → Dropout → Softmax(5).
6. **Entrenamiento**: Adam (1e-3), `sparse_categorical_crossentropy`, `EarlyStopping`, `ReduceLROnPlateau` y `ModelCheckpoint`.
7. **Evaluación en test**: accuracy, precisión/recall/F1 por clase y macro, y matriz de confusión (conteos y normalizada).

## 4. Instalación

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 5. Entrenar el modelo

**Opción A: notebook en Google Colab (recomendado, con GPU)**
Sube `notebook_experimentacion.ipynb`, activa GPU (*Entorno de ejecución → Cambiar tipo → T4*) y ejecútalo completo.
Al final descarga `modelo_proyecto.zip`, `reports_proyecto.zip` y `ejemplos_proyecto.zip` y descomprímelos
en la raíz del repositorio (deben quedar las carpetas `modelo/`, `reports/` y `ejemplos/`).

**Opción B: por consola**
```bash
python train.py                       # con data augmentation
python train.py --no-augmentation     # para comparar
python train.py --epochs 60 --batch-size 64
```
Genera `modelo/modelo_flores.keras`, `modelo/config.json`, las gráficas y métricas en `reports/` y 10 imágenes en `ejemplos/`.

## 6. Predecir por consola

```bash
python predict.py ejemplos/rosa_1.jpg ejemplos/tulipan_1.jpg
```

## 7. Ejecutar la aplicación Streamlit

```bash
streamlit run app.py
```

Abre http://localhost:8501, sube una imagen (JPG o PNG) y pulsa **Clasificar**. La app muestra la clase
predicha, la confianza y la probabilidad de las 5 clases. Si la confianza es menor al 60 % muestra un aviso.

> ⚠️ La app necesita `modelo/modelo_flores.keras`. Si ves "No se encontró el modelo", entrena primero (sección 5).

## 8. Entrega en Git (importante)

El modelo entrenado **debe estar en el repositorio**, o la app (y Streamlit Cloud) no podrá cargarlo:

```bash
git init
git add .
git commit -m "Proyecto CNN: clasificación de flores"
git branch -M main
git remote add origin https://github.com/TU_USUARIO/TU_REPO.git
git push -u origin main
```

Antes de hacer `git add`, comprueba que existan `modelo/modelo_flores.keras`, `modelo/config.json` y `reports/metrics.json`.
(Git no sube carpetas vacías.) Para desplegar en **Streamlit Community Cloud**, elige `app.py` como archivo principal y,
en *Advanced settings*, Python 3.11 o 3.12.

## 9. Resultados

Pega aquí tus resultados tras entrenar (están en `reports/metrics.json` y `reports/comparacion_experimentos.csv`):

| Experimento | Val. accuracy | Test accuracy | F1 macro |
|-------------|---------------|---------------|----------|
| Sin augmentation | | | |
| Con augmentation | | | |

## 10. Limitaciones y trabajo futuro

- El modelo solo conoce 5 clases: ante cualquier otra imagen igual elegirá una flor (por eso se muestra la confianza).
- Dataset pequeño y CNN entrenada desde cero; el rendimiento tiene un techo.
- Los menús propios de Streamlit (barra superior) y los registros de Keras durante el entrenamiento están en el idioma de esas librerías.
- Mejoras: *transfer learning* (MobileNetV2 / EfficientNet), mayor resolución, más datos y una clase "otro".
