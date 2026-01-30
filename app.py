# -*- coding: utf-8 -*-
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

MODELS_DIR = Path("models_bime")

st.set_page_config(page_title="Calculadora Bimekizumab PASI", layout="centered")
st.title("Calculadora predictiva - Bimekizumab (semana 12-16)")
st.caption("Herramienta de apoyo a la decisión clínica. No sustituye al juicio clínico.")

# Inputs (mismos del modelo)
pasi = st.number_input("PASI basal (PASI INICIO TTO)", 0.0, 80.0, 20.0, step=0.5)
edad = st.number_input("Edad (años)", 18, 100, 45, step=1)
imc = st.number_input("IMC", 15.0, 60.0, 27.0, step=0.1)
sexo = st.selectbox("Sexo", ["Varón", "Mujer"])
artritis_txt = st.selectbox("Artritis", ["No", "Sí"])
nprev = st.number_input("Nº biológicos previos", 0, 20, 0, step=1)

if st.button("Calcular probabilidad"):
    model_75 = joblib.load(MODELS_DIR / "bime_PASI75_w16.joblib")
    model_90 = joblib.load(MODELS_DIR / "bime_PASI90_w16.joblib")

    X = pd.DataFrame([{
        "PASI INICIO TTO": float(pasi),
        "Edad (autocálculo)": float(edad),
        "IMC (autocálculo)": float(imc),
        "Sexo": sexo,
        "Artritis": 1 if artritis_txt == "Sí" else 0,
        "N biológicos previos": int(nprev),
    }])

    X75 = X.reindex(columns=model_75.feature_names_in_)
    X90 = X.reindex(columns=model_90.feature_names_in_)

    prob75 = model_75.predict_proba(X75)[0, 1]
    prob90 = model_90.predict_proba(X90)[0, 1]

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Probabilidad PASI75 (semana 12-16)", f"{prob75*100:.1f}%")
    with col2:
        st.metric("Probabilidad PASI90 (semana 12-16)", f"{prob90*100:.1f}%")
