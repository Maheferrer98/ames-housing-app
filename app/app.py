# app.py
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
# Variables más importantes
# =========================
# Ajusta esta lista según tu análisis SHAP
top_10_features = model_columns[:10]  # O puedes reemplazar por la lista exacta de top 10

# Dividir numéricas y categóricas
numeric_features = [f for f in top_10_features if "_ " not in f and f in model_columns]
categorical_features = [f for f in top_10_features if f not in numeric_features]

# =========================
# Interfaz Streamlit
# =========================
st.title("Predicción de Precio de Vivienda - Ames Housing")

st.write("Introduce las características de la vivienda:")

# Diccionarios para guardar los inputs
input_dict = {}

# Inputs numéricos
for feature in numeric_features:
    min_val = 0
    max_val = 10000  # ajusta según feature real
    input_dict[feature] = st.number_input(feature, min_value=min_val, max_value=max_val, value=0)

# Inputs categóricos
for feature in categorical_features:
    # Extraer categorías del dataset original
    cat_cols = [c for c in model_columns if c.startswith(feature + "_")]
    options = [c.replace(feature + "_", "") for c in cat_cols]
    if options:
        selected = st.selectbox(f"{feature}", options)
        # Crear one-hot
        for opt in options:
            input_dict[f"{feature}_{opt}"] = 1 if opt == selected else 0
    else:
        # En caso de que no haya categorías disponibles
        st.warning(f"No hay opciones para {feature}")

# Botón de predicción
if st.button("Calcular precio"):
    input_df = pd.DataFrame([input_dict], columns=model_columns)
    
    try:
        log_price = pipeline.predict(input_df)[0]
        price = np.expm1(log_price)  # Inversa de log1p
        st.success(f"💰 Precio estimado: ${price:,.2f}")
    except Exception as e:
        st.error(f"Error al calcular el precio: {e}")

