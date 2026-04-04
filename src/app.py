"""
RootKit — Aplicación web de detección de enfermedades en cultivos.
Samsung Innovation Campus 2026 | Equipo 5: Carlos · Max · Othon

Archivo: src/app.py
Descripción: Aplicación Streamlit con los tres modelos integrados:
             CNN (clasificación), K-Means (agrupamiento), Regresión (severidad).

Dependencias de modelos (ruta relativa desde src/):
    ../models/best_cnn_v1.h5
    ../models/cnn_metadata.pkl
    ../models/clustering_kmeans.pkl
    ../models/clustering_pca.pkl
    ../models/clustering_scaler.pkl
    ../models/cluster_semantica.pkl
    ../models/regresion_severidad.pkl
"""

import os
import pickle
import cv2
import numpy as np
import streamlit as st
from PIL import Image

# TensorFlow importado con supresión de logs de nivel INFO/WARNING
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow.keras.models import load_model

# ---------------------------------------------------------------------------
# Configuración global de la página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RootKit — Detección de Enfermedades",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Rutas de artefactos
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

PATH_CNN       = os.path.join(MODELS_DIR, "best_cnn_v1.h5")
PATH_METADATA  = os.path.join(MODELS_DIR, "cnn_metadata.pkl")
PATH_KMEANS    = os.path.join(MODELS_DIR, "clustering_kmeans.pkl")
PATH_PCA       = os.path.join(MODELS_DIR, "clustering_pca.pkl")
PATH_SCALER    = os.path.join(MODELS_DIR, "clustering_scaler.pkl")
PATH_SEMANTICA = os.path.join(MODELS_DIR, "cluster_semantica.pkl")
PATH_REGRESION = os.path.join(MODELS_DIR, "regresion_severidad.pkl")

ASSETS_DIR = os.path.join(BASE_DIR, "..", "assets", "cluster_samples")

# Tamaño de entrada por defecto; se sobreescribe en cargar_modelos()
IMG_SIZE = (224, 224)

# ---------------------------------------------------------------------------
# Funciones auxiliares para carga de modelos subclasificados
# ---------------------------------------------------------------------------

def _inferir_input_size(cnn) -> tuple:
    """
    Determina la forma de entrada (H, W) del modelo.

    Estrategia 1: extraer batch_input_shape desde get_config().
    Estrategia 2: probar tamaños estándar con una pasada forward y
                  seleccionar el primero que no genere error de forma.

    Returns
    -------
    tuple (H, W)
    """
    try:
        cfg = cnn.get_config()
        layers_cfg = cfg.get("layers", [])
        if layers_cfg:
            first = layers_cfg[0]["config"]
            bis = first.get("batch_input_shape") or first.get("batch_shape")
            if bis and len(bis) >= 3 and bis[1] and bis[2]:
                return int(bis[1]), int(bis[2])
    except Exception:
        pass

    for size in [128, 64, 224, 32, 256]:
        try:
            _t = np.zeros((1, size, size, 3), dtype=np.float32)
            cnn(_t, training=False)
            return size, size
        except Exception:
            continue

    return 128, 128


def _construir_extractor(cnn):
    """
    Construye un extractor de características compatible con modelos
    subclasificados, Functional y Sequential.

    Para modelos subclasificados cnn.input no está definido, por lo que
    Model(inputs=cnn.input, ...) no es viable. Se encapsula la extracción
    en una clase que ejecuta cnn.layers[:-1] de forma secuencial,
    omitiendo la capa de clasificación softmax final.

    Returns
    -------
    tf.keras.Model con método call() que devuelve el vector de características.
    """
    capas_internas = cnn.layers[:-1]

    class ExtractorSecuencial(tf.keras.Model):
        """Ejecuta capas[:-1] del modelo base en secuencia."""

        def __init__(self, capas):
            super().__init__(name="feature_extractor")
            self._capas_base = capas

        def call(self, x, training=False):
            for capa in self._capas_base:
                try:
                    x = capa(x, training=training)
                except TypeError:
                    x = capa(x)
            return x

    return ExtractorSecuencial(capas_internas)


# ---------------------------------------------------------------------------
# Carga de modelos con caché (se ejecuta una sola vez por sesión)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Cargando modelos — esto ocurre una sola vez...")
def cargar_modelos():
    """
    Carga y devuelve todos los artefactos entrenados.

    Returns
    -------
    tuple
        (cnn, extractor, metadata, kmeans, pca, scaler, regresor,
         cluster_semantica)
    """
    # --- CNN ---
    cnn = load_model(PATH_CNN)

    _input_h, _input_w = _inferir_input_size(cnn)
    _dummy = np.zeros((1, _input_h, _input_w, 3), dtype=np.float32)
    cnn(_dummy, training=False)

    extractor = _construir_extractor(cnn)
    extractor._rootkit_input_hw = (_input_h, _input_w)

    # --- Metadata CNN ---
    with open(PATH_METADATA, "rb") as f:
        metadata = pickle.load(f)

    # --- Artefactos de clustering ---
    with open(PATH_KMEANS,    "rb") as f: kmeans            = pickle.load(f)
    with open(PATH_PCA,       "rb") as f: pca               = pickle.load(f)
    with open(PATH_SCALER,    "rb") as f: scaler            = pickle.load(f)
    with open(PATH_SEMANTICA, "rb") as f: cluster_semantica = pickle.load(f)

    # --- Modelo de regresión ---
    with open(PATH_REGRESION, "rb") as f: regresor = pickle.load(f)

    return cnn, extractor, metadata, kmeans, pca, scaler, regresor, cluster_semantica


# ---------------------------------------------------------------------------
# Preprocesamiento de imagen
# ---------------------------------------------------------------------------

def preprocesar_imagen(
    imagen_pil: Image.Image,
    img_size: tuple = IMG_SIZE,
) -> np.ndarray:
    """
    Redimensiona y normaliza una imagen PIL para inferencia CNN.

    Returns
    -------
    np.ndarray de forma (1, H, W, 3) con valores en [0, 1].
    """
    img = imagen_pil.resize((img_size[1], img_size[0]))  # PIL recibe (W, H)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


# ---------------------------------------------------------------------------
# Extracción de features HSV
# ---------------------------------------------------------------------------

RANGOS_PLANTAS = {
    'Tomato':  {'hoja': ([5,  20, 20], [95, 255, 255])},
    'Potato':  {'hoja': ([0,  10, 10], [90, 255, 255])},
    'Pepper':  {'hoja': ([25, 30, 30], [95, 255, 255])},
    'default': {'hoja': ([0,  20, 20], [100, 255, 255])},
}


def extraer_histograma_hsv(
    imagen_pil: Image.Image,
    bins: int = 32,
) -> np.ndarray:
    """
    Extrae un histograma HSV concatenado de la imagen.

    Pipeline idéntico al usado en el entrenamiento de clustering_scaler.pkl,
    clustering_pca.pkl y clustering_kmeans.pkl:
    32 bins × 3 canales (H, S, V) = 96 dimensiones, normalización L1.

    Returns
    -------
    np.ndarray de forma (1, 96).
    """
    img_rgb = np.array(imagen_pil.convert("RGB"), dtype=np.uint8)
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    histogramas = []
    rangos = [(0, 180), (0, 256), (0, 256)]  # H, S, V — rangos OpenCV
    for canal, (rng_min, rng_max) in enumerate(rangos):
        hist = cv2.calcHist(
            [img_hsv], [canal], None, [bins], [rng_min, rng_max]
        )
        hist = hist.flatten().astype(np.float32)
        total = hist.sum()
        if total > 0:
            hist /= total  # normalización L1
        histogramas.append(hist)

    return np.concatenate(histogramas).reshape(1, -1)  # (1, 96)


def extraer_features_regresion(imagen_pil: Image.Image, nombre_clase: str) -> np.ndarray:
    """
    Extrae 16 features HSV específicas para el modelo de regresión.

    Pipeline: máscara de hoja por rango HSV → histograma de Hue (16 bins)
    → normalización L1. Idéntico al pipeline del notebook de regresión.

    Returns
    -------
    np.ndarray de forma (1, 16).
    """
    img_rgb = np.array(imagen_pil.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    planta = nombre_clase.split('_')[0] if '_' in nombre_clase else nombre_clase.split(' ')[0]
    rango = RANGOS_PLANTAS.get(planta, RANGOS_PLANTAS['default'])

    lower_leaf = np.array(rango['hoja'][0])
    upper_leaf = np.array(rango['hoja'][1])
    mask_leaf  = cv2.inRange(hsv, lower_leaf, upper_leaf)

    hist_hue = cv2.calcHist([hsv], [0], mask_leaf, [16], [0, 180])
    total = hist_hue.sum()
    if total > 0:
        hist_hue /= total

    return hist_hue.flatten().reshape(1, -1)  # (1, 16)


# ---------------------------------------------------------------------------
# Resolución de nombres de clase CNN
# ---------------------------------------------------------------------------

PLANTVILLAGE_CLASES = [
    "Pepper bell — Bacterial spot",
    "Pepper bell — Healthy",
    "Potato — Early blight",
    "Potato — Late blight",
    "Potato — Healthy",
    "Tomato — Bacterial spot",
    "Tomato — Early blight",
    "Tomato — Late blight",
    "Tomato — Leaf mold",
    "Tomato — Septoria leaf spot",
    "Tomato — Spider mites",
    "Tomato — Target spot",
    "Tomato — Yellow leaf curl virus",
    "Tomato — Mosaic virus",
    "Tomato — Healthy",
]


def _resolver_class_names(metadata: dict, n_clases: int) -> list:
    """
    Extrae nombres de clases desde metadata.
    Prueba múltiples claves antes de usar el fallback canónico PlantVillage.
    """
    for clave in ("class_names", "classes", "labels", "label_names",
                  "class_labels", "idx_to_class", "names"):
        val = metadata.get(clave)
        if val is not None:
            if isinstance(val, (list, tuple)) and len(val) == n_clases:
                return list(val)
            if isinstance(val, dict):
                try:
                    return [val[i] for i in range(n_clases)]
                except KeyError:
                    pass
    if n_clases == len(PLANTVILLAGE_CLASES):
        return PLANTVILLAGE_CLASES
    return [f"Clase {i}" for i in range(n_clases)]


# ---------------------------------------------------------------------------
# Funciones de inferencia
# ---------------------------------------------------------------------------

def inferir_clasificacion(cnn, tensor: np.ndarray, metadata: dict) -> dict:
    """
    Ejecuta la clasificación CNN y devuelve etiqueta, confianza y top-5.

    Returns
    -------
    dict con claves: etiqueta, confianza, top5
    """
    probs      = cnn.predict(tensor, verbose=0)[0]
    class_names = _resolver_class_names(metadata, len(probs))

    idx_top   = int(np.argmax(probs))
    confianza = float(probs[idx_top])
    top5_idx  = np.argsort(probs)[::-1][:5]
    top5      = [(class_names[i], float(probs[i])) for i in top5_idx]

    return {"etiqueta": class_names[idx_top], "confianza": confianza, "top5": top5}


def inferir_clustering(imagen_pil: Image.Image, scaler, pca, kmeans) -> dict:
    """
    Extrae histograma HSV (96 dims), aplica la pipeline de clustering
    y devuelve el cluster asignado.

    Pipeline: HSV hist (96) → StandardScaler → PCA (50) → K-Means → cluster_id

    Returns
    -------
    dict con claves: cluster_id, n_clusters, features_pca
    """
    features   = extraer_histograma_hsv(imagen_pil)   # (1, 96)
    f_scaled   = scaler.transform(features)            # (1, 96)
    f_pca      = pca.transform(f_scaled)               # (1, 50)
    cluster_id = int(kmeans.predict(f_pca)[0])

    return {
        "cluster_id":   cluster_id,
        "n_clusters":   kmeans.n_clusters,
        "features_pca": f_pca,
    }


def inferir_regresion(imagen_pil: Image.Image, scaler, pca, regresor) -> dict:
    """
    Estima la severidad de la enfermedad.

    Pipeline: HSV (96) → StandardScaler → PCA → slice(n_reg) → regresor

    Returns
    -------
    dict con claves: severidad (float [0,1]), porcentaje (float [0,100])
    """
    import pandas as pd

    features = extraer_histograma_hsv(imagen_pil)
    f_scaled = scaler.transform(features)
    f_pca    = pca.transform(f_scaled)

    n_reg = getattr(regresor, "n_features_in_", f_pca.shape[1])
    f_reg = f_pca[:, :n_reg]

    feature_names = getattr(regresor, "feature_names_in_", None)
    if feature_names is not None:
        f_reg = pd.DataFrame(f_reg, columns=feature_names[:n_reg])

    severidad = float(np.clip(regresor.predict(f_reg)[0], 0.0, 1.0))

    return {"severidad": severidad, "porcentaje": round(severidad * 100, 2)}


# ---------------------------------------------------------------------------
# Utilidades de presentación
# ---------------------------------------------------------------------------

def descripcion_cluster(cluster_id: int, cluster_semantica: dict) -> tuple:
    """
    Devuelve (etiqueta, descripcion) desde el mapeo semántico generado en NB4.

    Returns
    -------
    tuple (etiqueta: str, descripcion: str)
    """
    entrada = cluster_semantica.get(cluster_id)
    if entrada:
        proporcion = entrada["proporcion_dominante"] * 100
        return (
            entrada["etiqueta"],
            f"{entrada['descripcion']} "
            f"(patrón predominante en {proporcion:.0f}% del grupo)",
        )
    return f"Grupo {cluster_id + 1}", "Patrón visual identificado por K-Means."


def nivel_severidad(porcentaje: float) -> tuple:
    """
    Devuelve nivel textual y clave de color Streamlit según el porcentaje.

    Returns
    -------
    tuple (nivel: str, color_streamlit: str)
    """
    if porcentaje < 15:
        return "Bajo", "success"
    elif porcentaje < 40:
        return "Moderado", "warning"
    elif porcentaje < 70:
        return "Alto", "error"
    else:
        return "Crítico", "error"


# ---------------------------------------------------------------------------
# Interfaz: barra lateral
# ---------------------------------------------------------------------------
st.sidebar.title("RootKit")
st.sidebar.markdown("**Samsung Innovation Campus 2026**")
st.sidebar.markdown("Equipo 5: Carlos · Max · Othon")
st.sidebar.markdown("---")

pagina = st.sidebar.radio(
    "Navegación",
    options=["Diagnóstico", "Acerca del proyecto"],
)

# ---------------------------------------------------------------------------
# Página principal: Diagnóstico
# ---------------------------------------------------------------------------
if pagina == "Diagnóstico":

    st.title("Sistema de Detección de Enfermedades en Cultivos")
    st.markdown(
        "Carga una imagen de hoja de **jitomate**, **papa** o **pimiento** "
        "para obtener el diagnóstico integrado de los tres modelos."
    )
    st.markdown("---")

    # Carga de modelos
    try:
        cnn, extractor, metadata, kmeans, pca, scaler, regresor, cluster_semantica = cargar_modelos()
        modelos_disponibles = True
    except Exception as exc:
        st.error(f"Error al cargar los modelos: {exc}")
        st.info(
            "Verifica que los archivos `.h5` y `.pkl` estén presentes en `models/` "
            "y que las rutas en `app.py` sean correctas."
        )
        modelos_disponibles = False

    # Carga de imagen
    imagen_cargada = st.file_uploader(
        label="Selecciona una imagen (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
    )

    if imagen_cargada is not None and modelos_disponibles:

        imagen    = Image.open(imagen_cargada).convert("RGB")
        _model_hw = getattr(extractor, "_rootkit_input_hw", IMG_SIZE)
        tensor    = preprocesar_imagen(imagen, img_size=_model_hw)

        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(imagen, caption="Imagen cargada", use_container_width=True)

        with col2:
            st.subheader("Resultados del diagnóstico")

            tab_clf, tab_clu, tab_reg = st.tabs(
                ["Clasificación (CNN)", "Agrupamiento (K-Means)", "Severidad (Regresión)"]
            )

            # ------------------------------------------------------------------
            # TAB 1 — Clasificación CNN
            # ------------------------------------------------------------------
            with tab_clf:
                with st.spinner("Ejecutando clasificación..."):
                    try:
                        res_clf = inferir_clasificacion(cnn, tensor, metadata)

                        st.metric(
                            label="Diagnóstico",
                            value=res_clf["etiqueta"],
                            delta=f"Confianza: {res_clf['confianza']*100:.1f} %",
                        )

                        st.markdown("**Top 5 predicciones**")
                        for nombre, prob in res_clf["top5"]:
                            st.progress(
                                float(prob),
                                text=f"{nombre}: {prob*100:.2f} %",
                            )

                    except Exception as exc:
                        st.error(f"Error en clasificación: {exc}")

            # ------------------------------------------------------------------
            # TAB 2 — Clustering K-Means
            # ------------------------------------------------------------------
            with tab_clu:
                with st.spinner("Ejecutando agrupamiento..."):
                    try:
                        res_clu = inferir_clustering(imagen, scaler, pca, kmeans)

                        etiqueta_cluster, desc_cluster = descripcion_cluster(
                            res_clu["cluster_id"], cluster_semantica
                        )

                        # Referencia explícita al diagnóstico autoritativo (CNN)
                        st.info(
                            f"El diagnóstico de referencia es: "
                            f"**{res_clf['etiqueta']}** "
                            f"(CNN · confianza {res_clf['confianza']*100:.1f} %). "
                            "Las imágenes a continuación corresponden al grupo con "
                            "mayor similitud visual en espacio de color HSV."
                        )

                        # Imágenes representativas del cluster
                        carpeta_cluster = os.path.join(
                            ASSETS_DIR, f"cluster_{res_clu['cluster_id']}"
                        )
                        if os.path.exists(carpeta_cluster):
                            imagenes_similares = sorted(
                                os.listdir(carpeta_cluster)
                            )[:4]
                            if imagenes_similares:
                                st.markdown("**Casos con patrón visual similar**")
                                cols = st.columns(4)
                                for col, nombre_img in zip(cols, imagenes_similares):
                                    ruta_img = os.path.join(carpeta_cluster, nombre_img)
                                    col.image(
                                        Image.open(ruta_img),
                                        use_container_width=True,
                                        caption="Caso similar",
                                    )

                    except Exception as exc:
                        st.error(f"Error en agrupamiento: {exc}")

            # ------------------------------------------------------------------
            # TAB 3 — Regresión de severidad
            # ------------------------------------------------------------------
            with tab_reg:
                with st.spinner("Estimando severidad..."):
                    try:
                        clase_actual       = res_clf["etiqueta"]
                        features_hsv       = extraer_features_regresion(imagen, clase_actual)
                        porcentaje_predicho = regresor.predict(features_hsv)[0]

                        nivel, color_str = nivel_severidad(porcentaje_predicho)

                        st.metric(
                            label="Severidad estimada",
                            value=f"{porcentaje_predicho:.2f} %",
                            delta=f"Nivel: {nivel}",
                        )

                        st.progress(
                            min(porcentaje_predicho / 100, 1.0),
                            text=f"Afectación: {porcentaje_predicho:.1f} %",
                        )

                        if color_str == "success":
                            st.success("Nivel bajo. Monitoreo preventivo.")
                        elif color_str == "warning":
                            st.warning("Nivel moderado. Aplicar tratamiento.")
                        else:
                            st.error("Nivel crítico. Intervención inmediata.")

                    except Exception as exc:
                        st.error(f"Error en regresión: {exc}")

    elif imagen_cargada is None:
        st.warning("Carga una imagen para iniciar el diagnóstico.")

# ---------------------------------------------------------------------------
# Página secundaria: Acerca del proyecto
# ---------------------------------------------------------------------------
elif pagina == "Acerca del proyecto":

    st.title("Acerca de RootKit")
    st.markdown("""
    **RootKit** es un sistema integral de detección automática de enfermedades
    en cultivos agrícolas, desarrollado como proyecto final del curso
    Samsung Innovation Campus 2026.

    ### Modelos implementados
    | Módulo | Algoritmo | Responsable | Artefacto |
    |---|---|---|---|
    | Clasificación | CNN (MobileNetV2) | Carlos | `best_cnn_v1.h5` |
    | Agrupamiento | K-Means + PCA | Othon | `clustering_kmeans.pkl` |
    | Regresión de severidad | Random Forest sobre features HSV | Max | `regresion_severidad.pkl` |

    ### Dataset
    PlantVillage — 15 clases (jitomate, papa, pimiento)

    ### Stack tecnológico
    Python 3.10 · TensorFlow / Keras · scikit-learn · Streamlit · Google Colab · GitHub

    ### Arquitectura del pipeline de inferencia
    ```
    Imagen RGB
         │
         ▼
    Preprocesamiento (resize + normalización [0, 1])
         │
         ├──► CNN completo ──────────────────────► Clasificación (15 clases)
         │
         └──► Histograma HSV (32 bins × 3 = 96)
                   │
                   ├──► StandardScaler → PCA (50) → K-Means ──► Grupo visual
                   │
                   └──► Máscara HSV → Hue hist (16) → Regresor ─► Severidad (%)
    ```
    """)