import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- Configuración de rutas ---
BASE_DIR = os.path.dirname(__file__)
pipeline_path = os.path.join(BASE_DIR, "ames_price_pipeline.pkl")
columns_path = os.path.join(BASE_DIR, "model_columns.pkl")
importance_path = os.path.join(BASE_DIR, "top_features.pkl")  # archivo con top 10

# --- Cargar pipeline, columnas y top features ---
pipeline = joblib.load(pipeline_path)
model_columns = joblib.load(columns_path)
top_features = joblib.load(importance_path)  # lista de 10 variables más importantes

st.title("Predicción de Precio de Vivienda - Ames Housing")
st.write("Introduce las características más importantes de la vivienda para obtener una predicción de precio.")

# --- Detectar columnas categóricas y numéricas dentro de las top 12 ---
categorical_cols = [c for c in top_features if "_" in c]  # One-hot encoded
numerical_cols = [c for c in top_features if c not in categorical_cols]

# --- Crear diccionarios para inputs ---
cat_dict = {}  # {base_var: [categorías]}
for c in categorical_cols:
    base = c.split("_")[0]
    if base not in cat_dict:
        cat_dict[base] = []
    cat_dict[base].append(c.split("_")[1])

st.header("Introduce los valores de la vivienda:")

user_input = {}

# --- Inputs dinámicos ---
for col in numerical_cols:
    if col != "SalePrice":
        user_input[col] = st.number_input(f"{col}", min_value=0, value=100)

for base_col, options in cat_dict.items():
    user_input[base_col] = st.selectbox(f"{base_col}", options)

# --- Convertir inputs a DataFrame ---
input_df = pd.DataFrame([user_input])

# --- One-Hot Encoding manual para columnas categóricas ---
for base_col, options in cat_dict.items():
    for opt in options:
        col_name = f"{base_col}_{opt}"
        input_df[col_name] = 1 if input_df[base_col][0] == opt else 0
    input_df.drop(columns=[base_col], inplace=True)

# --- Añadir columnas faltantes dentro de top 12 ---
for c in top_features:
    if c not in input_df.columns:
        input_df[c] = 0

# --- Reordenar columnas ---
input_df = input_df[top_features]

# --- Predicción ---
if st.button("Predecir precio"):
    log_price = pipeline.predict(input_df)[0]
    price = np.expm1(log_price)  # revertir log1p
    st.success(f"💰 Precio estimado: {price:,.2f} €")
