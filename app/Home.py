"""
Home.py - Dashboard de Predicción de Abandono de Clientes (Churn)
Colegio Universitario de Cartago - BD-151

Ejecutar:
    streamlit run app/Home.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(
    page_title="Churn Predictor | CUC",
    page_icon="📡",
    layout="wide",
)

st.title("📡 Sistema de Predicción de Abandono de Clientes (Churn)")
st.caption("Colegio Universitario de Cartago — BD-151 Inteligencia Artificial Aplicada — Proyecto B")

st.markdown(
    """
Panel de control paara la **predicción y análisis de abandono (churn)**
de clientes de telecomunicaciones, basado en redes neuronales artificiales (ANN).

### 🧭 Navegación
- **1_Prediccion**: predicción individual (Yes/No + score de riesgo) para un cliente.
- **2_Analisis**: análisis por lotes (carga de CSV) y segmentación de la base de clientes.
- **3_Metricas**: métricas de rendimiento de los modelos y análisis de ROI de retención.
"""
)

st.divider()

DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", " processed", "customer_clean.csv"
)

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)

    col1, col2, col3, col4 = st.columns(4)
    total = len(df)
    churners = int(df["Churn"].sum())
    churn_rate = churners / total * 100
    avg_tenure = df["tenure"].mean()

    col1.metric("👥 Total de clientes", f"{total:,}")
    col2.metric("🚪 Clientes con churn", f"{churners:,}")
    col3.metric("📉 Tasa de churn", f"{churn_rate:.1f}%")
    col4.metric("📅 Antigüedad promedio", f"{avg_tenure:.1f} meses")

    st.subheader("Distribución general de Churn")
    c1, c2 = st.columns(2)

    with c1:
        churn_counts = df["Churn"].value_counts().rename({0: "No Churn", 1: "Churn"})
        fig = px.pie(
            names=churn_counts.index,
            values=churn_counts.values,
            title="Proporción de clientes que abandonan",
            color=churn_counts.index,
            color_discrete_map={"No Churn": "#2E86AB", "Churn": "#E74C3C"},
            hole=0.45,
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig2 = px.histogram(
            df, x="tenure", color=df["Churn"].map({0: "No Churn", 1: "Churn"}),
            nbins=30, barmode="overlay",
            title="Antigüedad (tenure) vs Churn",
            color_discrete_map={"No Churn": "#2E86AB", "Churn": "#E74C3C"},
            labels={"color": "Churn"},
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Cargos mensuales vs Churn")
    fig3 = px.box(
        df, x=df["Churn"].map({0: "No Churn", 1: "Churn"}), y="MonthlyCharges",
        color=df["Churn"].map({0: "No Churn", 1: "Churn"}),
        title="Distribución de cargos mensuales por grupo de churn",
        color_discrete_map={"No Churn": "#2E86AB", "Churn": "#E74C3C"},
        labels={"x": "Grupo", "MonthlyCharges": "Cargo mensual (USD)"},
    )
    st.plotly_chart(fig3, use_container_width=True)

else:
    st.info(
        "No se encontró el archivo de datos procesados "
        f"(`{DATA_PATH}`). Las métricas generales no se mostrarán, "
        "pero las páginas de predicción funcionan igual si la API está activa."
    )


