# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- Configurar rutas ---
BASE_DIR = os.path.dirname(__file__)
pipeline_path = os.path.join(BASE_DIR, "ames_price_pipeline.pkl")
columns_path = os.path.join(BASE_DIR, "model_columns.pkl")

# --- Cargar modelo y columnas ---
pipeline = joblib.load(pipeline_path)
model_columns = joblib.load(columns_path)

st.title("Predicción de Precio de Vivienda - Ames Housing")

st.write("""
Introduce las características de la vivienda para obtener una predicción de precio.
""")

# --- Lista de las 10 variables más importantes según SHAP ---
important_vars = [
    "Overall Qual", "Gr Liv Area", "Garage Cars", "Total Bsmt SF",
    "1st Flr SF", "Full Bath", "Year Built", "TotRms AbvGrd",
    "Fireplaces", "Lot Area"
]

# Diccionario para almacenar inputs
user_input = {}

# --- Generar inputs dinámicos según tipo ---
for col in important_vars:
    # Detectar si es categórica según columnas del modelo
    cat_cols = [c for c in model_columns if c.startswith(col + "_")]
    if cat_cols:
        # Dropdown con todas las categorías
        options = [c.replace(col + "_", "") for c in cat_cols]
        user_input[col] = st.selectbox(f"{col}", options)
    else:
        # Numérico: input numérico
        user_input[col] = st.number_input(f"{col}", min_value=0, value=100)

# --- Convertir inputs a DataFrame ---
input_df = pd.DataFrame([user_input])

# --- One-Hot Encoding manual para que coincida con el modelo ---
for col in input_df.columns:
    # Si es categórica
    cat_cols = [c for c in model_columns if c.startswith(col + "_")]
    if cat_cols:
        # Crear columnas dummy
        for dummy in cat_cols:
            val = dummy.replace(col + "_", "")
            input_df[dummy] = 1 if input_df[col][0] == val else 0
        input_df.drop(columns=[col], inplace=True)

# --- Añadir columnas faltantes en caso de que falten ---
for c in model_columns:
    if c not in input_df.columns:
        input_df[c] = 0

# --- Reordenar columnas ---
input_df = input_df[model_columns]

# --- Botón para predecir ---
if st.button("Predecir precio"):
    # Predecir
    log_price = pipeline.predict(input_df)[0]
    price = np.expm1(log_price)  # revertir log1p
    st.success(f"💰 Precio estimado: {price:,.2f} €")
