from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch

from model import RedBoston
from sqlalchemy import text
from database import engine
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy import MetaData, Table

metadata = MetaData()

housing_observations = Table(
    "housing_observations",
    metadata,
    autoload_with=engine
)

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



st.subheader("Guardar observación en PostgreSQL")

st.write(
    "Registra las características de la vivienda y, "
    "si lo conoces, su valor real."
)

with st.form("registro_vivienda"):

    medv_real = st.number_input(
        "Valor real (miles de USD). Usa 0 si es desconocido.",
        min_value=0.0,
        value=0.0,
        step=0.5
    )

    guardar = st.form_submit_button(
        "Guardar observación"
    )

if guardar:

    datos = {
        "crim": entradas["CRIM"],
        "zn": entradas["ZN"],
        "indus": entradas["INDUS"],
        "chas": int(entradas["CHAS"]),
        "nox": entradas["NOX"],
        "rm": entradas["RM"],
        "age": entradas["AGE"],
        "dis": entradas["DIS"],
        "rad": int(entradas["RAD"]),
        "tax": entradas["TAX"],
        "ptratio": entradas["PTRATIO"],
        "b": entradas["B"],
        "lstat": entradas["LSTAT"],
        "medv": medv_real if medv_real > 0 else None,
        "source": "streamlit_manual"
    }

    consulta = text("""
        INSERT INTO housing_observations (
            crim, zn, indus, chas, nox, rm, age,
            dis, rad, tax, ptratio, b, lstat,
            medv, source
        )
        VALUES (
            :crim, :zn, :indus, :chas, :nox, :rm, :age,
            :dis, :rad, :tax, :ptratio, :b, :lstat,
            :medv, :source
        )
        RETURNING id
    """)

    try:
        with engine.begin() as conexion:
            nuevo_id = conexion.execute(
                consulta, datos
            ).scalar_one()

        st.success(
            f"Observación {nuevo_id} guardada correctamente."
        )

    except Exception as error:
        st.error(
            "No se pudo guardar la observación. "
            "Revisa la conexión y los valores introducidos."
        )
        st.exception(error)

        st.divider()
st.subheader("📂 Carga masiva de observaciones")

archivo = st.file_uploader(
    "Selecciona un archivo CSV o Excel",
    type=["csv", "xlsx"]
)

columnas_modelo = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX",
    "RM", "AGE", "DIS", "RAD", "TAX",
    "PTRATIO", "B", "LSTAT"
]

if archivo is not None:

    try:
        if archivo.name.lower().endswith(".csv"):
            df_carga = pd.read_csv(archivo)
        else:
            df_carga = pd.read_excel(archivo)

        # Normalizar nombres de columnas
        df_carga.columns = (
            df_carga.columns.astype(str)
            .str.strip()
            .str.upper()
        )

        faltantes = [
            col for col in columnas_modelo
            if col not in df_carga.columns
        ]

        if faltantes:
            st.error(
                f"Faltan columnas obligatorias: {faltantes}"
            )

        else:
            if "MEDV" not in df_carga.columns:
                df_carga["MEDV"] = np.nan

            columnas_validas = columnas_modelo + ["MEDV"]

            df_carga = df_carga[columnas_validas].copy()

            for col in columnas_validas:
                df_carga[col] = pd.to_numeric(
                    df_carga[col],
                    errors="coerce"
                )

            # No aceptar características incompletas
            incompletos = df_carga[
                columnas_modelo
            ].isna().any(axis=1)

            # Validaciones básicas de dominio
            invalidos = (
                ~df_carga["CHAS"].isin([0, 1])
                | (df_carga["RM"] <= 0)
                | (df_carga["NOX"] <= 0)
                | (df_carga["MEDV"].notna() & (df_carga["MEDV"] < 0))
                | (df_carga["RAD"] % 1 != 0)
            )

            rechazados = incompletos | invalidos

            df_validos = df_carga.loc[
                ~rechazados
            ].copy()

            # Eliminar duplicados exactos dentro del archivo
            df_validos = df_validos.drop_duplicates()

            st.write("Vista previa de datos válidos")
            st.dataframe(df_validos.head(20))

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Filas recibidas",
                len(df_carga)
            )

            col2.metric(
                "Filas rechazadas",
                int(rechazados.sum())
            )

            col3.metric(
                "Filas listas para cargar",
                len(df_validos)
            )

            if st.button(
                "Guardar archivo en PostgreSQL",
                disabled=df_validos.empty
            ):
                df_sql = df_validos.rename(
                    columns=str.lower
                ).copy()

                df_sql["source"] = (
                    "upload_" + archivo.name[:90]
                )

                # Los registros nuevos quedan sin verificar
                df_sql["is_verified"] = False

                tabla = housing_observations

                registros = (
                    df_sql.astype(object)
                    .where(pd.notna(df_sql), None)
                    .to_dict(orient="records")
                )

                try:
                    total_insertados = 0

                    with engine.begin() as conexion:
                        for inicio in range(0, len(registros), 500):
                            lote = registros[inicio:inicio + 500]

                            consulta = insert(tabla).values(lote)

                            consulta = consulta.on_conflict_do_nothing(
                                index_elements=[
                                    "crim", "zn", "indus", "chas",
                                    "nox", "rm", "age", "dis",
                                    "rad", "tax", "ptratio", "b",
                                    "lstat", "medv"
                                ]
                            ).returning(tabla.c.id)

                            resultado = conexion.execute(consulta)

                            ids_insertados = resultado.scalars().all()

                            total_insertados += len(ids_insertados)

                    duplicados = len(registros) - total_insertados

                    st.success(
                        f"Registros insertados: {total_insertados}"
                    )

                    st.info(
                        f"Registros omitidos por duplicidad: {duplicados}"
                    )

                except Exception as error:
                    st.error("Error durante la inserción.")
                    st.exception(error)

    except Exception as error:
        st.error("Error durante la lectura o validación del archivo.")
        st.exception(error)