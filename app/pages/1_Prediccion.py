"""
1_Prediccion.py - Predicción individual de churn para un cliente.
"""

import streamlit as st
import requests
import plotly.graph_objects as go

st.set_page_config(page_title="Predicción Individual | Churn", page_icon="🔮", layout="wide")

API_URL = st.sidebar.text_input("URL de la API", value="http://127.0.0.1:8000")

st.title("🔮 Predicción Individual de Churn")
st.markdown("Completa los datos del cliente para obtener la predicción de abandono y su nivel de riesgo.")

with st.form("formulario_cliente"):
    st.subheader("Datos demográficos")
    c1, c2, c3, c4 = st.columns(4)
    gender = c1.selectbox("Género", ["Female", "Male"])
    senior = c2.selectbox("Adulto mayor (Senior Citizen)", ["No", "Sí"])
    partner = c3.selectbox("Tiene pareja", ["No", "Sí"])
    dependents = c4.selectbox("Tiene dependientes", ["No", "Sí"])

    st.subheader("Información de cuenta")
    c1, c2, c3 = st.columns(3)
    tenure = c1.number_input("Antigüedad (meses)", min_value=0, max_value=100, value=12)
    monthly_charges = c2.number_input("Cargo mensual (USD)", min_value=0.0, value=70.0, step=1.0)
    total_charges = c3.number_input("Cargo total acumulado (USD)", min_value=0.0, value=840.0, step=10.0)

    c1, c2 = st.columns(2)
    contract = c1.selectbox("Tipo de contrato", ["Month-to-month", "One year", "Two year"])
    payment_method = c2.selectbox(
        "Método de pago",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    )
    paperless = st.selectbox("Facturación sin papel (Paperless Billing)", ["No", "Sí"])

    st.subheader("Servicios contratados")
    c1, c2, c3, c4 = st.columns(4)
    phone_service = c1.selectbox("Servicio telefónico", ["No", "Sí"])
    multiple_lines = c2.selectbox("Múltiples líneas", ["No", "Sí"])
    internet_service = c3.selectbox("Servicio de internet", ["DSL", "Fiber optic", "No"])
    online_security = c4.selectbox("Seguridad en línea", ["No", "Sí"])

    c1, c2, c3, c4 = st.columns(4)
    online_backup = c1.selectbox("Respaldo en línea", ["No", "Sí"])
    device_protection = c2.selectbox("Protección de dispositivo", ["No", "Sí"])
    tech_support = c3.selectbox("Soporte técnico", ["No", "Sí"])
    streaming_tv = c4.selectbox("Streaming TV", ["No", "Sí"])

    streaming_movies = st.selectbox("Streaming Movies", ["No", "Sí"])

    submitted = st.form_submit_button("Predecir", type="primary")

if submitted:
    def yn(v):
        return 1 if v == "Sí" else 0

    payload = {
        "gender": 1 if gender == "Male" else 0,
        "SeniorCitizen": yn(senior),
        "Partner": yn(partner),
        "Dependents": yn(dependents),
        "tenure": int(tenure),
        "PhoneService": yn(phone_service),
        "MultipleLines": yn(multiple_lines),
        "OnlineSecurity": yn(online_security),
        "OnlineBackup": yn(online_backup),
        "DeviceProtection": yn(device_protection),
        "TechSupport": yn(tech_support),
        "StreamingTV": yn(streaming_tv),
        "StreamingMovies": yn(streaming_movies),
        "PaperlessBilling": yn(paperless),
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges),
        "Contract_Month-to-month": 1 if contract == "Month-to-month" else 0,
        "Contract_One year": 1 if contract == "One year" else 0,
        "Contract_Two year": 1 if contract == "Two year" else 0,
        "PaymentMethod_Bank transfer (automatic)": 1 if payment_method == "Bank transfer (automatic)" else 0,
        "PaymentMethod_Credit card (automatic)": 1 if payment_method == "Credit card (automatic)" else 0,
        "PaymentMethod_Electronic check": 1 if payment_method == "Electronic check" else 0,
        "PaymentMethod_Mailed check": 1 if payment_method == "Mailed check" else 0,
        "InternetService_DSL": 1 if internet_service == "DSL" else 0,
        "InternetService_Fiber optic": 1 if internet_service == "Fiber optic" else 0,
        "InternetService_No": 1 if internet_service == "No" else 0,
    }

    try:
        r_churn = requests.post(f"{API_URL}/predict/churn", json=payload, timeout=10)
        r_risk = requests.post(f"{API_URL}/predict/risk_score", json=payload, timeout=10)

        if r_churn.status_code == 200 and r_risk.status_code == 200:
            churn_data = r_churn.json()
            risk_data = r_risk.json()

            st.divider()
            st.subheader("Resultados")

            col1, col2, col3 = st.columns(3)

            with col1:
                pred = churn_data["churn_prediction"]
                color = "#E74C3C" if pred == "Yes" else "#2ECC71"
                st.markdown(
                    f"<div style='padding:1rem;border-radius:10px;background-color:{color}22;border:2px solid {color}'>"
                    f"<h4>Predicción de Churn</h4>"
                    f"<h2 style='color:{color}'>{'⚠️ Abandonará' if pred == 'Yes' else '✅ Se mantiene'}</h2>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            with col2:
                st.metric("Probabilidad de Churn", f"{churn_data['churn_probability']*100:.1f}%")
                st.metric("Confianza del modelo", f"{churn_data['confidence']*100:.1f}%")

            with col3:
                level = risk_data["risk_level"]
                level_colors = {"Bajo": "#2ECC71", "Medio": "#F39C12", "Alto": "#E74C3C"}
                st.markdown(
                    f"<div style='padding:1rem;border-radius:10px;background-color:{level_colors[level]}22;border:2px solid {level_colors[level]}'>"
                    f"<h4>Nivel de Riesgo</h4>"
                    f"<h2 style='color:{level_colors[level]}'>{level}</h2>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            st.divider()

            c1, c2 = st.columns([1, 1.2])

            with c1:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=risk_data["risk_score"] * 100,
                    title={"text": "Score de Riesgo de Churn (%)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#34495E"},
                        "steps": [
                            {"range": [0, 33], "color": "#2ECC71"},
                            {"range": [33, 66], "color": "#F39C12"},
                            {"range": [66, 100], "color": "#E74C3C"},
                        ],
                    },
                ))
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                st.markdown("#### 💡 Recomendación de acción")
                st.info(risk_data["recommended_action"])

                st.markdown("#### 📋 Resumen de la predicción")
                st.json({
                    "churn_prediction": churn_data["churn_prediction"],
                    "churn_probability": churn_data["churn_probability"],
                    "risk_score": risk_data["risk_score"],
                    "risk_level": risk_data["risk_level"],
                })
        else:
            st.error(f"Error en la API: {r_churn.text} / {r_risk.text}")

    except requests.exceptions.ConnectionError:
        st.error(
            "No se pudo conectar con la API. Asegúrate de que esté corriendo en "
            f"`{API_URL}` (ejecuta: `uvicorn api.main:app --reload`)."
        )
