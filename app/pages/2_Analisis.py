"""
2_Analisis.py - Análisis por lotes: carga un CSV de clientes, obtiene
predicciones de churn y risk_score para cada uno mediante la API,
y segmenta la base de clientes por nivel de riesgo.
"""

import streamlit as st
import pandas as pd
import requests
import plotly.express as px

st.set_page_config(page_title="Análisis por Lotes | Churn", page_icon="📊", layout="wide")

API_URL = st.sidebar.text_input("URL de la API", value="http://127.0.0.1:8000")

st.title("📊 Análisis de Cartera de Clientes")
st.markdown(
    """
Sube un archivo **CSV** con clientes en el mismo formato que `customer_clean.csv`
(las mismas 26 columnas usadas para entrenar los modelos, sin la columna `Churn`).
La aplicación enviará cada fila a la API y generará una segmentación por
nivel de riesgo.
"""
)

REQUIRED_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
    "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "PaperlessBilling",
    "MonthlyCharges", "TotalCharges",
    "Contract_Month-to-month", "Contract_One year", "Contract_Two year",
    "PaymentMethod_Bank transfer (automatic)", "PaymentMethod_Credit card (automatic)",
    "PaymentMethod_Electronic check", "PaymentMethod_Mailed check",
    "InternetService_DSL", "InternetService_Fiber optic", "InternetService_No",
]

uploaded_file = st.file_uploader("Cargar archivo CSV de clientes", type=["csv"])

max_rows = st.slider("Máximo de filas a procesar (para no saturar la API)", 10, 1000, 200, 10)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        st.error(f"Faltan columnas requeridas en el CSV: {missing}")
    else:
        df = df.head(max_rows)
        st.success(f"Archivo cargado: {len(df)} clientes (limitado a {max_rows}).")

        if st.button("🚀 Ejecutar predicciones", type="primary"):
            results = []
            progress = st.progress(0)
            for i, row in df.iterrows():
                payload = row[REQUIRED_COLUMNS].to_dict()
                try:
                    r_risk = requests.post(f"{API_URL}/predict/risk_score", json=payload, timeout=10)
                    r_churn = requests.post(f"{API_URL}/predict/churn", json=payload, timeout=10)
                    risk = r_risk.json()
                    churn = r_churn.json()
                    results.append({
                        "churn_prediction": churn["churn_prediction"],
                        "churn_probability": churn["churn_probability"],
                        "risk_score": risk["risk_score"],
                        "risk_level": risk["risk_level"],
                        "recommended_action": risk["recommended_action"],
                    })
                except requests.exceptions.ConnectionError:
                    st.error(
                        "No se pudo conectar con la API. Asegúrate de que esté "
                        f"corriendo en `{API_URL}`."
                    )
                    st.stop()
                progress.progress((i + 1) / len(df))

            results_df = pd.DataFrame(results)
            final_df = pd.concat([df.reset_index(drop=True), results_df], axis=1)
            st.session_state["analisis_resultados"] = final_df

if "analisis_resultados" in st.session_state:
    final_df = st.session_state["analisis_resultados"]

    st.divider()
    st.subheader("Resultados")

    col1, col2, col3, col4 = st.columns(4)
    total = len(final_df)
    en_riesgo = (final_df["churn_prediction"] == "Yes").sum()
    col1.metric("Total analizados", total)
    col2.metric("En riesgo de churn", int(en_riesgo))
    col3.metric("% en riesgo", f"{en_riesgo/total*100:.1f}%")
    col4.metric("Score promedio", f"{final_df['risk_score'].mean():.2f}")

    c1, c2 = st.columns(2)
    with c1:
        level_counts = final_df["risk_level"].value_counts().reindex(["Bajo", "Medio", "Alto"]).fillna(0)
        fig = px.bar(
            x=level_counts.index, y=level_counts.values,
            color=level_counts.index,
            color_discrete_map={"Bajo": "#2ECC71", "Medio": "#F39C12", "Alto": "#E74C3C"},
            title="Segmentación de clientes por nivel de riesgo",
            labels={"x": "Nivel de riesgo", "y": "Número de clientes"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig2 = px.histogram(
            final_df, x="risk_score", nbins=20,
            title="Distribución de scores de riesgo",
            color_discrete_sequence=["#34495E"],
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Clientes prioritarios (riesgo Alto)")
    alto_riesgo = final_df[final_df["risk_level"] == "Alto"].sort_values("risk_score", ascending=False)
    st.dataframe(
        alto_riesgo[["tenure", "MonthlyCharges", "TotalCharges", "churn_prediction", "risk_score", "risk_level", "recommended_action"]],
        use_container_width=True,
    )

    st.download_button(
        "⬇️ Descargar resultados (CSV)",
        data=final_df.to_csv(index=False).encode("utf-8"),
        file_name="resultados_churn.csv",
        mime="text/csv",
    )
