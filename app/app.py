import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ======================
# 1. Cargar modelo y columnas
pipeline = joblib.load("ames_price_pipeline.pkl")
model_columns = joblib.load("model_columns.pkl")

# ======================
# 2. Título de la app
st.title("Predicción de precio de viviendas (Ames Housing)")

st.markdown("Introduce las características de la vivienda:")

# ======================
# 3. Inputs para 10 variables más importantes
# Aquí pon tus 10 variables más importantes según SHAP o RF
# Ejemplo:
numeric_vars = {
    "Gr Liv Area": (500, 4000, 1500),
    "Total Bsmt SF": (0, 3000, 800),
    "Garage Cars": (0, 4, 2),
    "Overall Qual": (1, 10, 5),
    "House_Age": (0, 150, 20),
    "Since_Remod": (0, 100, 10),
    "Garage_Age": (0, 100, 10),
    "Lot Area": (500, 20000, 5000)
}

categorical_vars = {
    "Neighborhood": ["NAmes", "CollgCr", "OldTown", "Edwards", "Somerst", "Gilbert", "NWAmes", "Sawyer", "SawyerW", "Mitchel"], 
    "BldgType": ["1Fam", "2fmCon", "Duplex", "TwnhsE", "Twnhs"]
}

# ======================
# 3.1 Entradas numéricas
input_data = {}
for var, (min_val, max_val, default) in numeric_vars.items():
    input_data[var] = st.number_input(var, min_value=min_val, max_value=max_val, value=default)

# 3.2 Entradas categóricas
for var, options in categorical_vars.items():
    input_data[var] = st.selectbox(var, options)

# ======================
# 4. Convertir a DataFrame y codificar
df_input = pd.DataFrame([input_data])

# One-Hot Encoding
df_input_encoded = pd.get_dummies(df_input)

# Asegurar todas las columnas del modelo
df_input_encoded = df_input_encoded.reindex(columns=model_columns, fill_value=0)

# ======================
# 5. Predicción
if st.button("Predecir precio"):
    pred_log = pipeline.predict(df_input_encoded)[0]
    pred_real = np.expm1(pred_log)
    st.success(f"💰 Precio estimado: ${pred_real:,.2f}")
