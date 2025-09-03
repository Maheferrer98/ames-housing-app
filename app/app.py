import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Cargar pipeline y columnas
pipeline = joblib.load("ames_price_pipeline.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("Predicción de Precio de Vivienda - Ames Housing")

st.markdown("Introduce las características de la vivienda:")

# --- VARIABLES NUMÉRICAS ---
GrLivArea = st.number_input("Superficie habitable (sq ft)", min_value=300, max_value=6000, value=1500)
TotalBsmtSF = st.number_input("Superficie sótano (sq ft)", min_value=0, max_value=3000, value=800)
GarageCars = st.number_input("Plazas de garaje", min_value=0, max_value=5, value=2)
OverallQual = st.number_input("Calidad general (1-10)", min_value=1, max_value=10, value=6)
HouseAge = st.number_input("Edad de la vivienda (años)", min_value=0, max_value=150, value=30)
SinceRemod = st.number_input("Años desde remodelación", min_value=0, max_value=100, value=10)
GarageAge = st.number_input("Edad del garaje (años)", min_value=0, max_value=100, value=20)

# --- VARIABLES CATEGÓRICAS ---
Neighborhood = st.selectbox("Vecindario", ["NAmes", "CollgCr", "OldTown", "Edwards", "Somerst"])
BldgType = st.selectbox("Tipo de edificio", ["1Fam", "2fmCon", "Duplex", "TwnhsE"])
HouseStyle = st.selectbox("Estilo de casa", ["1Story", "2Story", "1.5Fin", "SLvl"])
RoofStyle = st.selectbox("Estilo de techo", ["Gable", "Hip", "Flat"])
CentralAir = st.selectbox("Aire acondicionado central", ["Y", "N"])

# Botón de predicción
if st.button("Calcular precio"):

    # Crear dict con input del usuario
    input_data = {
        "Gr Liv Area": GrLivArea,
        "Total Bsmt SF": TotalBsmtSF,
        "Garage Cars": GarageCars,
        "Overall Qual": OverallQual,
        "House_Age": HouseAge,
        "Since_Remod": SinceRemod,
        "Garage_Age": GarageAge,
        # Categóricas en One-Hot
        f"Neighborhood_{Neighborhood}": 1,
        f"BldgType_{BldgType}": 1,
        f"HouseStyle_{HouseStyle}": 1,
        f"RoofStyle_{RoofStyle}": 1,
        f"CentralAir_{CentralAir}": 1
    }

    # Crear DataFrame con todas las columnas que espera el modelo
    input_df = pd.DataFrame([input_data])
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[model_columns]

    # Predicción
    log_price = pipeline.predict(input_df)[0]
    price = np.expm1(log_price)

    st.success(f"💰 Precio estimado: ${price:,.2f}")
