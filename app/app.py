import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Cargar modelo
pipeline = joblib.load("ames_pipeline_ct.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("Predicción de Precio de Vivienda Ames")

st.write("Introduce las características de la vivienda:")

# Sliders para variables numéricas
numeric_inputs = {}
for col in model_columns:
    if col != "Neighborhood":
        min_val = int(pipeline.named_steps["preprocessor"].named_transformers_["num"].named_steps["imputer"].statistics_[numeric_features.index(col)])
        max_val = int(df[col].max())
        default_val = int(df[col].median())
        numeric_inputs[col] = st.number_input(f"{col}", min_value=min_val, max_value=max_val, value=default_val)

# Dropdown para vecindario
neighborhoods = df["Neighborhood"].unique().tolist()
categorical_input = st.selectbox("Neighborhood", neighborhoods)

# Crear DataFrame para predicción
input_df = pd.DataFrame({**numeric_inputs, "Neighborhood": [categorical_input]})

if st.button("Calcular Precio"):
    log_price = pipeline.predict(input_df)[0]
    price = np.expm1(log_price)
    st.success(f"💰 Precio estimado: ${price:,.2f}")
