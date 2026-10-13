
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import torch

from model import RedBoston


st.set_page_config(
    page_title="Boston Housing | ML Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

FEATURE_DEFAULTS = {
    "CRIM": 0.1,
    "ZN": 0.0,
    "INDUS": 8.0,
    "CHAS": 0,
    "NOX": 0.5,
    "RM": 6.0,
    "AGE": 60.0,
    "DIS": 4.0,
    "RAD": 4,
    "TAX": 300.0,
    "PTRATIO": 18.0,
    "B": 390.0,
    "LSTAT": 12.0
}

DESCRIPCIONES = {
    "CRIM": "Tasa de criminalidad per cápita",
    "ZN": "Porcentaje de suelo residencial de baja densidad",
    "INDUS": "Porcentaje de suelo comercial no minorista",
    "CHAS": "Proximidad al río Charles",
    "NOX": "Concentración de óxidos de nitrógeno",
    "RM": "Número promedio de habitaciones",
    "AGE": "Porcentaje de viviendas antiguas",
    "DIS": "Distancia ponderada a centros de empleo",
    "RAD": "Índice de accesibilidad a autopistas",
    "TAX": "Tasa de impuesto a la propiedad",
    "PTRATIO": "Relación alumnos/profesor",
    "B": "Variable histórica derivada de composición racial",
    "LSTAT": "Porcentaje histórico de población de menor estatus socioeconómico"
}


@st.cache_resource
def cargar_modelo():
    scaler_x = joblib.load(MODELS_DIR / "scaler_X.pkl")
    scaler_y = joblib.load(MODELS_DIR / "scaler_y.pkl")

    modelo = RedBoston(len(scaler_x.feature_names_in_))
    modelo.load_state_dict(
        torch.load(
            MODELS_DIR / "boston_nn.pth",
            map_location="cpu",
            weights_only=True
        )
    )
    modelo.eval()

    return modelo, scaler_x, scaler_y


modelo, scaler_x, scaler_y = cargar_modelo()
variables = list(scaler_x.feature_names_in_)

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    max-width: 1250px;
}
div[data-testid="stMetric"] {
    background: rgba(128, 128, 128, 0.08);
    border: 1px solid rgba(128, 128, 128, 0.15);
    padding: 18px;
    border-radius: 12px;
}
div[data-testid="stMetricValue"] {
    font-size: 1.8rem;
}
</style>
""", unsafe_allow_html=True)


# Sidebar
with st.sidebar:
    st.title("🏠 Boston Housing")
    st.caption("Neural Network · PyTorch")
    st.divider()

    st.subheader("Configuración de vivienda")
    st.caption("Modifica los valores para generar una estimación.")

    entradas = {}

    for variable in variables:
        if variable == "CHAS":
            entradas[variable] = st.selectbox(
                "CHAS · Río Charles",
                options=[0, 1],
                help=DESCRIPCIONES[variable]
            )
        elif variable == "RAD":
            entradas[variable] = st.number_input(
                variable,
                min_value=0,
                value=int(FEATURE_DEFAULTS[variable]),
                step=1,
                help=DESCRIPCIONES[variable]
            )
        else:
            entradas[variable] = st.number_input(
                variable,
                value=float(FEATURE_DEFAULTS[variable]),
                format="%.3f",
                help=DESCRIPCIONES[variable]
            )

    predecir = st.button(
        "Generar predicción",
        type="primary",
        use_container_width=True
    )


# Header
st.caption("MACHINE LEARNING / REGRESSION / PYTORCH")
st.title("Property Value Intelligence")
st.write(
    "Dashboard interactivo para explorar estimaciones históricas "
    "del conjunto Boston Housing mediante una red neuronal."
)

st.divider()

# Model performance
st.subheader("Rendimiento del modelo")

m1, m2, m3, m4 = st.columns(4)

m1.metric("R² Test", "0.8519")
m2.metric("MAE", "2.0069", help="Miles de USD")
m3.metric("RMSE", "3.2952", help="Miles de USD")
m4.metric("Variables", "13")

st.caption(
    "Métricas obtenidas durante la evaluación original del modelo "
    "sobre 102 observaciones de prueba."
)

st.divider()

# Prediction
st.subheader("Estimación del valor histórico")

if predecir:
    datos = pd.DataFrame([entradas], columns=variables)
    datos_scaled = scaler_x.transform(datos)

    tensor = torch.tensor(
        datos_scaled,
        dtype=torch.float32
    )

    with torch.no_grad():
        pred_scaled = modelo(tensor).numpy()

    prediccion = float(
        scaler_y.inverse_transform(pred_scaled)[0, 0]
    )

    st.session_state["ultima_prediccion"] = prediccion

if "ultima_prediccion" in st.session_state:
    prediccion = st.session_state["ultima_prediccion"]

    st.metric(
        "Valor estimado",
        f"${prediccion * 1000:,.2f} USD"
    )

    st.info(
        "Estimación educativa basada en datos históricos. "
        "No representa una tasación inmobiliaria actual."
    )
else:
    st.info(
        "Configura las características en la barra lateral "
        "y selecciona 'Generar predicción'."
    )

st.divider()

# Features
st.subheader("Características del inmueble")

resumen = pd.DataFrame({
    "Variable": variables,
    "Valor": [entradas[v] for v in variables],
    "Descripción": [DESCRIPCIONES[v] for v in variables]
})

st.dataframe(
    resumen,
    use_container_width=True,
    hide_index=True
)

with st.expander("Metodología y limitaciones"):
    st.markdown("""
    **Modelo:** red neuronal multicapa (MLP) desarrollada en PyTorch.

    **Arquitectura:** 13 entradas → 64 neuronas → 32 neuronas → 1 salida.

    **Preprocesamiento:** estandarización de variables con
    `StandardScaler`.

    **Evaluación:** 102 observaciones reservadas para prueba.

    **Variable objetivo:** `MEDV`, expresada en miles de dólares
    históricos.

    **Limitaciones:** Boston Housing es un conjunto de datos
    antiguo y contiene variables socialmente sensibles,
    particularmente `B` y `LSTAT`. Esta aplicación se presenta
    exclusivamente con fines educativos y no debe utilizarse
    para decisiones inmobiliarias, crediticias o de vivienda.
    """)

st.caption(
    "Boston Housing ML Dashboard · Python · PyTorch · "
    "Scikit-learn · Streamlit"
)
