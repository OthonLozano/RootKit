"""
RootKit — Aplicación web de detección de enfermedades en cultivos.
Samsung Innovation Campus 2026 | Equipo 5: Carlos · Max · Othon

Archivo: src/app.py
Descripción: Aplicación Streamlit con los tres modelos integrados:
             CNN (clasificación), K-Means (agrupamiento), Regresión (severidad).

Dependencias de modelos (ruta relativa desde src/):
    ../models/cnn_v1.h5
    ../models/cnn_metadata.pkl
    ../models/clustering_kmeans.pkl
    ../models/clustering_pca.pkl
    ../models/clustering_scaler.pkl
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
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(BASE_DIR, "..", "models")

PATH_CNN        = os.path.join(MODELS_DIR, "best_cnn_v1.h5")#cnn_v1.h5
PATH_METADATA   = os.path.join(MODELS_DIR, "cnn_metadata.pkl")
PATH_KMEANS     = os.path.join(MODELS_DIR, "clustering_kmeans.pkl")
PATH_PCA        = os.path.join(MODELS_DIR, "clustering_pca.pkl")
PATH_SCALER     = os.path.join(MODELS_DIR, "clustering_scaler.pkl")
PATH_REGRESION  = os.path.join(MODELS_DIR, "regresion_severidad.pkl")

# Tamaño de entrada esperado por el CNN.
# Valor por defecto; puede ser sobreescrito en tiempo de ejecución
# una vez que cargar_modelos() determine el tamaño real del modelo.
IMG_SIZE = (224, 224)

# ---------------------------------------------------------------------------
# Carga de modelos con caché (se ejecuta una sola vez por sesión)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Funciones auxiliares para carga de modelos subclasificados
# ---------------------------------------------------------------------------

def _inferir_input_size(cnn) -> tuple:
    """
    Intenta determinar la forma de entrada (H, W) del modelo.

    Estrategia 1: extraer batch_input_shape desde get_config().
    Estrategia 2: probar tamaños estándar con una pasada forward y
                  seleccionar el primero que no genere error de forma.

    Returns
    -------
    tuple (H, W)
    """
    # Estrategia 1 — configuración serializada
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

    # Estrategia 2 — búsqueda por prueba
    for size in [128, 64, 224, 32, 256]:
        try:
            _t = np.zeros((1, size, size, 3), dtype=np.float32)
            cnn(_t, training=False)
            return size, size
        except Exception:
            continue

    # Fallback final
    return 128, 128


def _construir_extractor(cnn):
    """
    Construye un extractor de características compatible con modelos
    subclasificados, Functional y Sequential.

    Para modelos subclasificados, cnn.input no está definido, por lo que
    Model(inputs=cnn.input, ...) no es viable. En su lugar se devuelve un
    tf.keras.Model que ejecuta todas las capas del modelo base excepto la
    última (capa de clasificación softmax).

    Parameters
    ----------
    cnn : tf.keras.Model
        Modelo completo previamente llamado al menos una vez (grafo trazado).

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

        def call(self, x, training=False):  # noqa: D401
            for capa in self._capas_base:
                # Algunas capas aceptan training, otras no.
                try:
                    x = capa(x, training=training)
                except TypeError:
                    x = capa(x)
            return x

    return ExtractorSecuencial(capas_internas)
    
@st.cache_resource(show_spinner="Cargando modelos — esto ocurre una sola vez...")
def cargar_modelos():
    """
    Carga y devuelve todos los artefactos entrenados.

    Returns
    -------
    tuple
        (cnn_completo, extractor_features, metadata, kmeans, pca, scaler, regresor)
    """
    # --- CNN ---
    cnn = load_model(PATH_CNN)

    # -------------------------------------------------------------------------
    # Determinar la forma de entrada real del modelo.
    #
    # El modelo es de tipo subclasificado (CNN_PlantVillage_v1), por lo que
    # cnn.input no existe como atributo. Se infiere la forma de entrada desde
    # la configuración serializada del modelo; si no está disponible se prueban
    # tamaños estándar hasta encontrar el que no produce error de forma.
    # -------------------------------------------------------------------------
    _input_h, _input_w = _inferir_input_size(cnn)

    # Materializar el grafo con una pasada forward sobre el tensor de entrada
    # del tamaño correcto.
    _dummy = np.zeros((1, _input_h, _input_w, 3), dtype=np.float32)
    cnn(_dummy, training=False)

    # -------------------------------------------------------------------------
    # Extractor de características para modelos subclasificados.
    #
    # Model(inputs=cnn.input, ...) es incompatible con modelos subclasificados
    # porque cnn.input nunca se define en esa arquitectura.
    #
    # Se encapsula la extracción en una función que ejecuta cnn.layers[:-1]
    # de forma secuencial, omitiendo la capa de clasificación final.
    # -------------------------------------------------------------------------
    extractor = _construir_extractor(cnn)
    extractor._rootkit_input_hw = (_input_h, _input_w)

    # --- Metadata del CNN (nombres de clases, índice → etiqueta) ---
    with open(PATH_METADATA, "rb") as f:
        metadata = pickle.load(f)

    # --- Artefactos de clustering ---
    with open(PATH_KMEANS,  "rb") as f: kmeans  = pickle.load(f)
    with open(PATH_PCA,     "rb") as f: pca     = pickle.load(f)
    with open(PATH_SCALER,  "rb") as f: scaler  = pickle.load(f)

    # --- Modelo de regresión ---
    with open(PATH_REGRESION, "rb") as f: regresor = pickle.load(f)

    return cnn, extractor, metadata, kmeans, pca, scaler, regresor


# ---------------------------------------------------------------------------
# Preprocesamiento de imagen
# ---------------------------------------------------------------------------
def preprocesar_imagen(
    imagen_pil: Image.Image,
    img_size: tuple[int, int] = IMG_SIZE,
) -> np.ndarray:
    """
    Redimensiona y normaliza una imagen PIL para inferencia CNN.

    Parameters
    ----------
    imagen_pil : PIL.Image.Image
        Imagen cargada por el usuario (RGB).
    img_size : tuple[int, int]
        Tamaño (H, W) al que se redimensiona la imagen. Por defecto IMG_SIZE;
        se sobreescribe con el tamaño real del modelo una vez cargado.

    Returns
    -------
    np.ndarray
        Tensor de forma (1, H, W, 3) con valores en [0, 1].
    """
    img = imagen_pil.resize((img_size[1], img_size[0]))   # PIL: (W, H)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)



RANGOS_PLANTAS = {
    'Tomato': {'hoja': ([5, 20, 20], [95, 255, 255]), 'sano': ([25, 25, 25], [95, 255, 255])},
    'Potato': {'hoja': ([0, 10, 10], [90, 255, 255]), 'sano': ([38, 50, 50], [90, 255, 255])},
    'Pepper': {'hoja': ([25, 30, 30], [95, 255, 255]), 'sano': ([35, 50, 50], [95, 255, 255])},
    'default': {'hoja': ([0, 20, 20], [100, 255, 255]), 'sano': ([35, 40, 40], [90, 255, 255])}
}

def extraer_histograma_hsv(image_pil):
    # Convertir de PIL a OpenCV (BGR)
    img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    
    # Rangos que usaste en tu Notebook para segmentar la hoja
    bajo_hoja = np.array([30, 20, 20])
    alto_hoja = np.array([90, 255, 255])
    
    # Crear máscara y extraer histograma de Hue (16 bins como en el notebook)
    mascara_hoja = cv2.inRange(hsv, bajo_hoja, alto_hoja)
    hist_hue = cv2.calcHist([hsv], [0], mascara_hoja, [16], [0, 180])
    
    # Normalizar para que coincida con el entrenamiento
    cv2.normalize(hist_hue, hist_hue, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    
    return hist_hue.flatten().reshape(1, -1)


def extraer_features_regresion(imagen_pil, nombre_clase):
    # 1. Convertir de PIL a OpenCV
    img_rgb = np.array(imagen_pil.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # 2. Obtener rangos según la planta (Usa el diccionario RANGOS_PLANTAS)
    rango = RANGOS_PLANTAS.get(nombre_clase.split('_')[0], RANGOS_PLANTAS['default'])
    lower_leaf = np.array(rango['hoja'][0])
    upper_leaf = np.array(rango['hoja'][1])
    
    # 3. Crear máscara de la hoja
    mask_leaf = cv2.inRange(hsv, lower_leaf, upper_leaf)
    
    # 4. Extraer solo el histograma de HUE (16 bins) - IGUAL QUE EL NOTEBOOK
    hist_hue = cv2.calcHist([hsv], [0], mask_leaf, [16], [0, 180])
    
    # Normalizar
    total = hist_hue.sum()
    if total > 0:
        hist_hue /= total
        
    return hist_hue.flatten().reshape(1, -1) # Retorna (1, 16)


    

def extraer_histograma_hsv(
    imagen_pil: Image.Image,
    bins: int = 32,
) -> np.ndarray:
    """
    Extrae un histograma HSV concatenado de la imagen.

    Este es el mismo pipeline de features usado durante el entrenamiento
    de clustering_scaler.pkl, clustering_pca.pkl, clustering_kmeans.pkl
    y regresion_severidad.pkl (32 bins × 3 canales = 96 dimensiones).

    Parameters
    ----------
    imagen_pil : PIL.Image.Image
        Imagen RGB cargada por el usuario.
    bins : int
        Número de bins por canal. Debe coincidir con el valor usado en
        el notebook de entrenamiento (default: 32).

    Returns
    -------
    np.ndarray de forma (1, bins * 3) normalizado a [0, 1].
    """
    import cv2
    img_rgb = np.array(imagen_pil.convert("RGB"), dtype=np.uint8)
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    histogramas = []
    rangos = [(0, 180), (0, 256), (0, 256)]   # H, S, V — rangos OpenCV
    for canal, (rng_min, rng_max) in enumerate(rangos):
        hist = cv2.calcHist(
            [img_hsv], [canal], None, [bins], [rng_min, rng_max]
        )
        hist = hist.flatten().astype(np.float32)
        total = hist.sum()
        if total > 0:
            hist /= total   # normalización L1
        histogramas.append(hist)

    features = np.concatenate(histogramas).reshape(1, -1)  # (1, 96)
    return features

PLANTVILLAGE_CLASES = [
    "Pepper bell — Bacterial spot",       # 0
    "Pepper bell — Healthy",              # 1
    "Potato — Early blight",              # 2
    "Potato — Late blight",               # 3
    "Potato — Healthy",                   # 4
    "Tomato — Bacterial spot",            # 5
    "Tomato — Early blight",              # 6
    "Tomato — Late blight",               # 7
    "Tomato — Leaf mold",                 # 8
    "Tomato — Septoria leaf spot",        # 9
    "Tomato — Spider mites",              # 10
    "Tomato — Target spot",               # 11
    "Tomato — Yellow leaf curl virus",    # 12
    "Tomato — Mosaic virus",              # 13
    "Tomato — Healthy",                   # 14
]

def _resolver_class_names(metadata: dict, n_clases: int) -> list:
    """
    Intenta extraer nombres de clases desde el diccionario metadata.
    Prueba múltiples claves posibles antes de usar el fallback canónico.
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
    # Fallback: nombres canónicos PlantVillage si el conteo coincide
    if n_clases == len(PLANTVILLAGE_CLASES):
        return PLANTVILLAGE_CLASES
    # Último recurso: etiquetas numéricas
    return [f"Clase {i}" for i in range(n_clases)]

# ---------------------------------------------------------------------------
# Funciones de inferencia por módulo
# ---------------------------------------------------------------------------
def inferir_clasificacion(cnn, tensor: np.ndarray, metadata: dict) -> dict:
    """
    Ejecuta la clasificación CNN y devuelve etiqueta, confianza y top-5.

    Parameters
    ----------
    cnn : keras.Model
        Modelo de clasificación completo.
    tensor : np.ndarray
        Imagen preprocesada (1, 224, 224, 3).
    metadata : dict
        Diccionario con clave 'class_names': list[str].

    Returns
    -------
    dict con claves: etiqueta, confianza, top5
    """
    probs = cnn.predict(tensor, verbose=0)[0]
    class_names = _resolver_class_names(metadata, len(probs))

    idx_top   = int(np.argmax(probs))
    confianza = float(probs[idx_top])

    top5_idx  = np.argsort(probs)[::-1][:5]
    top5      = [(class_names[i], float(probs[i])) for i in top5_idx]

    return {
        "etiqueta":  class_names[idx_top],
        "confianza": confianza,
        "top5":      top5,
    }


def inferir_clustering(imagen_pil: Image.Image, scaler, pca, kmeans) -> dict:
    """
    Extrae histograma HSV, aplica la pipeline de clustering y asigna cluster.

    El clustering_scaler.pkl y clustering_pca.pkl fueron entrenados sobre
    histogramas HSV (32 bins × 3 canales = 96 dimensiones), no sobre
    features CNN. Se usa extraer_histograma_hsv() para mantener consistencia
    con el espacio de features del entrenamiento.

    Parameters
    ----------
    imagen_pil : PIL.Image.Image
        Imagen original cargada por el usuario.
    scaler, pca, kmeans : sklearn artifacts
        Pipeline de normalización → reducción → agrupamiento.

    Returns
    -------
    dict con claves: cluster_id, n_clusters, features_pca (np.ndarray 2-D)
    """
    features   = extraer_histograma_hsv(imagen_pil)           # (1, 96)
    f_scaled   = scaler.transform(features)                   # (1, 96)
    f_pca      = pca.transform(f_scaled)                      # (1, n_components)
    cluster_id = int(kmeans.predict(f_pca)[0])

    return {
        "cluster_id":   cluster_id,
        "n_clusters":   kmeans.n_clusters,
        "features_pca": f_pca,
    }


def inferir_regresion(imagen_pil: Image.Image, scaler, pca, regresor) -> dict:
    """
    Estima la severidad de la enfermedad.

    Pipeline: HSV (96) → StandardScaler → PCA → slice(n_reg) → RandomForest

    El PCA de clustering tiene n_components=50. El RandomForestRegressor de Max
    fue entrenado con n_features_in_ componentes PCA (valor leído en tiempo de
    ejecución desde regresor.n_features_in_). Dado que PCA ordena los vectores
    propios por varianza explicada descendente, tomar los primeros n_reg
    componentes preserva la mayor parte de la información y mantiene la
    compatibilidad dimensional sin requerir un artefacto adicional.

    Parameters
    ----------
    imagen_pil : PIL.Image.Image
        Imagen original cargada por el usuario.
    scaler : sklearn.StandardScaler
        Normalizador ajustado sobre los 96 features HSV.
    pca : sklearn.PCA
        Reductor ajustado con n_components=50 (clustering).
    regresor : sklearn estimator
        RandomForestRegressor; su atributo n_features_in_ define cuántos
        componentes PCA se le deben suministrar.

    Returns
    -------
    dict con claves: severidad (float en [0, 1]), porcentaje (float en [0, 100])
    """
    import pandas as pd

    features  = extraer_histograma_hsv(imagen_pil)            # (1, 96)
    f_scaled  = scaler.transform(features)                    # (1, 96)
    f_pca     = pca.transform(f_scaled)                       # (1, 50)

    # Determinar cuántos componentes espera el regresor y adaptar.
    n_reg = getattr(regresor, "n_features_in_", f_pca.shape[1])
    f_reg = f_pca[:, :n_reg]                                  # (1, n_reg)

    # Si el regresor fue entrenado con un DataFrame de pandas (feature names),
    # encapsular en DataFrame para suprimir el UserWarning y garantizar
    # compatibilidad con versiones futuras de scikit-learn.
    feature_names = getattr(regresor, "feature_names_in_", None)
    if feature_names is not None:
        f_reg = pd.DataFrame(f_reg, columns=feature_names[:n_reg])

    severidad = float(regresor.predict(f_reg)[0])
    severidad = float(np.clip(severidad, 0.0, 1.0))

    return {
        "severidad":   severidad,
        "porcentaje":  round(severidad * 100, 2),
    }


# ---------------------------------------------------------------------------
# Utilidades de presentación
# ---------------------------------------------------------------------------
ETIQUETAS_CLUSTER = {
    0: "Patrón de lesión: manchas necróticas",
    1: "Patrón de lesión: clorosis foliar",
    2: "Patrón de lesión: micelio superficial",
    3: "Patrón de lesión: decoloración vascular",
    4: "Patrón de lesión: tejido sano",
}

def descripcion_cluster(cluster_id: int, n_clusters: int) -> str:
    """Devuelve una descripción semántica del cluster asignado."""
    return ETIQUETAS_CLUSTER.get(
        cluster_id,
        f"Grupo visual {cluster_id + 1} de {n_clusters} "
        "(patrón identificado por K-Means)",
    )


def nivel_severidad(porcentaje: float) -> tuple[str, str]:
    """
    Devuelve nivel textual y color Streamlit según el porcentaje de severidad.

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
        "Carga una imagen de hoja de **tomate**, **papa** o **pimiento** "
        "para obtener el diagnóstico integrado de los tres modelos."
    )
    st.markdown("---")

    # Carga de modelos
    try:
        cnn, extractor, metadata, kmeans, pca, scaler, regresor = cargar_modelos()
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

        imagen = Image.open(imagen_cargada).convert("RGB")
        # Usar el tamaño de entrada real del modelo (inferido en cargar_modelos)
        _model_hw = getattr(extractor, "_rootkit_input_hw", IMG_SIZE)
        tensor = preprocesar_imagen(imagen, img_size=_model_hw)

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
                        res_clu = inferir_clustering(
                            imagen, scaler, pca, kmeans
                        )

                        st.metric(
                            label="Cluster asignado",
                            value=f"Cluster {res_clu['cluster_id']}",
                        )
                        st.info(
                            descripcion_cluster(
                                res_clu["cluster_id"],
                                res_clu["n_clusters"],
                            )
                        )

                        st.caption(
                            f"El modelo K-Means operó con {res_clu['n_clusters']} "
                            f"grupos y PCA de {pca.n_components_} componentes. "
                            "El cluster refleja el patrón visual predominante "
                            "independientemente de la especie vegetal."
                        )

                    except Exception as exc:
                        st.error(f"Error en agrupamiento: {exc}")

            # ------------------------------------------------------------------
            # TAB 3 — Regresión de severidad
            # ------------------------------------------------------------------
            with tab_reg:
                with st.spinner("Estimando severidad..."):
                    try:
                        # 1. USAMOS LA ETIQUETA QUE DIO LA CLASIFICACIÓN (RES_CLF VIENE DEL TAB 1)
                        clase_actual = res_clf["etiqueta"]
            
                        # 2. LLAMAMOS A TU NUEVA FUNCIÓN (LA QUE PUSISTE ARRIBA)
                        # Esto genera 16 features. NO USAMOS NI SCALER NI PCA AQUÍ.
                        features_hsv = extraer_features_regresion(imagen, clase_actual)
            
                        # 3. PREDICCIÓN DIRECTA CON TU MODELO .PKL
                        # Como tu modelo espera 16 y le damos 16, ya no habrá error.
                        porcentaje_predicho = regresor.predict(features_hsv)[0]
            
                        # 4. OBTENER NIVEL TEXTUAL (BAJO, ALTO, ETC.)
                        nivel, color_str = nivel_severidad(porcentaje_predicho)
            
                        # 5. MOSTRAR RESULTADOS
                        st.metric(
                            label="Severidad estimada",
                            value=f"{porcentaje_predicho:.2f} %",
                            delta=f"Nivel: {nivel}",
                        )
            
                        # Barra de progreso
                        st.progress(min(porcentaje_predicho/100, 1.0), text=f"Afectación: {porcentaje_predicho:.1f}%")
            
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
    | Clasificación | CNN (MobileNetV2) | Carlos | `best_cnn_v2.h5` |
    | Agrupamiento | K-Means + PCA | Othon | `clustering_kmeans.pkl` |
    | Regresión de severidad | Regresión sobre features CNN | Max | `regresion_severidad.pkl` |

    ### Dataset
    PlantVillage — 15 clases (tomate, papa, pimiento)

    ### Stack tecnológico
    Python 3.10 · TensorFlow / Keras · scikit-learn · Streamlit · Google Colab · GitHub

    ### Arquitectura del pipeline de inferencia
    ```
    Imagen (224×224 RGB)
         │
         ▼
    Preprocesamiento (normalización [0,1])
         │
         ├──► CNN completo ──────────────► Clasificación (15 clases)
         │
         └──► CNN extractor (features)
                   │
                   ├──► StandardScaler ──► PCA ──► K-Means ──► Cluster ID
                   │
                   └──► StandardScaler ──► Regresor ──────────► Severidad (%)
    ```
    """)