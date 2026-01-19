import streamlit as st
import joblib
import pandas as pd
from datetime import datetime

# Cargar el modelo (sube el archivo .joblib al mismo lugar que este app.py)
model = joblib.load('modelo_flightontime_xgboost_final.joblib')

st.title("✈️ FlightOnTime - Predicción de Retrasos")

st.write("Ingresa los datos del vuelo y descubre si hay riesgo de retraso.")

with st.form("form_prediccion"):
    aerolinea = st.text_input("Aerolínea (código IATA)", value="WN")
    origen = st.text_input("Aeropuerto origen (IATA)", value="ATL")
    destino = st.text_input("Aeropuerto destino (IATA)", value="DFW")
    fecha_partida = st.text_input("Fecha y hora de partida (YYYY-MM-DDTHH:MM:SS)", value="2026-01-15T18:00:00")
    distancia_km = st.number_input("Distancia en km", min_value=30.0, max_value=8000.0, value=1175.0)
    
    submitted = st.form_submit_button("Predecir")

if submitted:
    try:
        # Procesar la fecha
        dt = datetime.fromisoformat(fecha_partida.replace('T', ' '))
        hora = dt.hour
        dia = dt.weekday()
        pico = 1 if hora >= 15 else 0
        
        # Crear entrada para el modelo
        input_df = pd.DataFrame([{
            'AIRLINE_CODE': aerolinea,
            'ORIGIN': origen,
            'DEST': destino,
            'DISTANCE': distancia_km,
            'HORA_PARTIDA': hora,
            'DIA_SEMANA': dia,
            'ES_HORA_PICO': pico
        }])
        
        # Predicción
        prob = model.predict_proba(input_df)[0][1]
        prevision = "Retrasado" if prob > 0.5 else "Puntual"
        
        st.success(f"**Previsión:** {prevision}")
        st.metric("Probabilidad de retraso", f"{prob:.2%}")
        
    except Exception as e:
        st.error(f"Error al procesar: {str(e)}")
