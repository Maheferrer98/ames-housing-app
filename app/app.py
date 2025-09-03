import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# =========================
# Cargar pipeline y columnas
# =========================
current_dir = os.path.dirname(__file__)
pipeline_path = os.path.join(current_dir, "ames_price_pipeline.pkl")
columns_path = os.path.join(current_dir, "model_columns.pkl")

pipeline = joblib.load(pipeline_path)
model_columns = joblib.load(columns_path)

# =========================
# Variables seleccionadas
# =========================
# Top 9 numéricas + Neighborhood
numeric_features = {
    "Overall Qual": {"min": 1, "max": 10, "step": 1, "default": 5},
    "Gr Liv Area": {"min": 300, "max": 5000, "step": 10, "default": 1500},
    "Total Bsmt SF": {"min": 0, "max": 4000, "step": 10, "default": 1000},
    "Garage Cars": {"min": 0, "max": 5, "step": 1, "default": 2},
    "Garage Area": {"min": 0, "max": 1500, "step": 10, "default": 500},
    "1st Flr SF": {"min": 300, "max": 3000, "step": 10, "default": 1200},
    "Full Bath": {"min": 0, "max": 4, "step": 1, "default": 2},
    "TotRms AbvGrd": {"min": 2, "max": 15, "step": 1, "default": 6},
    "Year Built": {"min": 1870, "max": 2020, "step": 1, "default": 1970}
}

# Categoría
categorical_feature = "Neighborhood"

# =========================
# Interfaz Streamlit
# =========================
st.title("Predicción de Precio de Vivienda - Ames Housing")
st.write("Introduce las características de la vivienda:")

# Inputs numéricos
input_dict = {}
for feature, params in numeric_features.items():
    input_dict[feature] = st.number_input(
        feature,
        min_value=params["min"],
        max_value=params["max"],
        value=params["default"],
        step=params["step"]
    )

# Inputs categóricos
# Extraer opciones de model_columns
cat_cols = [c for c in model_columns if c.startswith(categorical_feature + "_")]
options = [c.replace(categorical_feature + "_", "") for c in cat_cols]

if options:
    selected = st.selectbox(f"{categorical_feature}", options)
    for opt in options:
        input_dict[f"{categorical_feature}_{opt}"] = 1 if opt == selected else 0
else:
    st.warning(f"No hay opciones para {categorical_feature}")

# =========================
# Predicción
# =========================
if st.button("Calcular precio"):
    input_df = pd.DataFrame([input_dict], columns=model_columns)
    try:
        log_price = pipeline.predict(input_df)[0]
        price = np.expm1(log_price)
        st.success(f"💰 Precio estimado: ${price:,.2f}")
    except Exception as e:
        st.error(f"Error al calcular el precio: {e}")
