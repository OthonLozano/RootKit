"""
RootKit — Aplicación web de detección de enfermedades en cultivos.
Samsung Innovation Campus 2026 | Equipo 5: Carlos · Max · Othon

Archivo: src/app.py
Descripción: Estructura base de la aplicación Streamlit con navegación,
             carga de imagen y placeholders para los tres modelos.
"""

import streamlit as st
from PIL import Image
import numpy as np

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
# Barra lateral de navegación
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

    # --- Carga de imagen ---
    imagen_cargada = st.file_uploader(
        label="Selecciona una imagen (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
    )

    if imagen_cargada is not None:
        imagen = Image.open(imagen_cargada).convert("RGB")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(imagen, caption="Imagen cargada", use_column_width=True)

        with col2:
            st.subheader("Resultados del diagnóstico")

            tab_clasificacion, tab_clustering, tab_regresion = st.tabs(
                ["Clasificación (CNN)", "Agrupamiento (K-Means)", "Severidad (Regresión)"]
            )

            with tab_clasificacion:
                st.info("Modelo CNN no integrado aún. Pendiente: clasificacion_cnn.h5 de Carlos.")

            with tab_clustering:
                st.info("Modelo K-Means no integrado aún. Pendiente: clustering_kmeans.pkl de Othon.")

            with tab_regresion:
                st.info("Modelo de regresión no integrado aún. Pendiente: regresion_severidad.pkl de Max.")

    else:
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
    - **Clasificación**: Red Neuronal Convolucional (MobileNetV2) — Carlos
    - **Agrupamiento**: K-Means sobre embeddings CNN con PCA — Othon
    - **Regresión de severidad**: Regresión sobre features CNN — Max

    ### Dataset
    PlantVillage — 15 clases (tomate, papa, pimiento)

    ### Stack tecnológico
    Python 3.10 · TensorFlow/Keras · scikit-learn · Streamlit · Google Colab · GitHub
    """)