from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch

from model import RedBoston


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"


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


st.set_page_config(
    page_title="Boston Housing | Neural Network",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 Boston Housing — Predicción con red neuronal")
st.write(
    "Aplicación educativa desarrollada con PyTorch y Streamlit."
)

modelo, scaler_x, scaler_y = cargar_modelo()

# Variables en el mismo orden utilizado durante el entrenamiento
variables = list(scaler_x.feature_names_in_)

# Valores iniciales ilustrativos
valores = {
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

st.subheader("Características de la vivienda")

col1, col2, col3 = st.columns(3)

entradas = {}

for i, variable in enumerate(variables):
    columna = [col1, col2, col3][i % 3]

    with columna:
        if variable == "CHAS":
            entradas[variable] = st.selectbox(
                "CHAS — Cercanía al río Charles",
                options=[0, 1]
            )
        else:
            entradas[variable] = st.number_input(
                variable,
                value=float(valores[variable])
            )

if st.button("Predecir precio", type="primary"):

    datos = pd.DataFrame(
        [entradas],
        columns=variables
    )

    datos_scaled = scaler_x.transform(datos)

    tensor = torch.tensor(
        datos_scaled,
        dtype=torch.float32
    )

    with torch.no_grad():
        pred_scaled = modelo(tensor).numpy()

    prediccion = scaler_y.inverse_transform(
        pred_scaled
    )[0, 0]

    st.metric(
        "Valor estimado histórico",
        f"${prediccion * 1000:,.2f} USD"
    )

    st.caption(
        "Estimación educativa basada en Boston Housing. "
        "No representa precios inmobiliarios actuales."
    )