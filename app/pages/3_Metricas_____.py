"""
3_Metricas.py - Métricas de rendimiento de los modelos ANN y simulador
de ROI para estrategias de retención de clientes.

Las métricas se ingresan manualmente a partir de los resultados obtenidos
al ejecutar el notebook 03_Entrenar_y_Guardar_Modelos.ipynb.
El simulador de ROI permite estimar el retorno de una campaña de retención
ajustando los parámetros de costo, tasa de éxito e ingresos esperados.

Proyecto B: Sistema de Predicción de Abandono de Clientes (Churn)
BD-151 Inteligencia Artificial Aplicada
Colegio Universitario de Cartago - 2026
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests

st.set_page_config(page_title="Métricas y ROI | Churn", page_icon="📈", layout="wide")

API_URL = st.sidebar.text_input("URL de la API", value="http://127.0.0.1:8000")

st.title("📈 Métricas de Modelos y Análisis de ROI")
st.caption("BD-151 Inteligencia Artificial Aplicada — Colegio Universitario de Cartago")

# ── Estado de la API ───────────────────────────────────────────────────────────
try:
    health = requests.get(f"{API_URL}/health", timeout=5).json()
    st.success(f"API conectada — estado: **{health['status']}**")
    c1, c2 = st.columns(2)
    c1.metric("Modelo de Clasificación", "✅ Cargado" if health["model_classification_loaded"] else "❌ No cargado")
    c2.metric("Modelo de Regresión",     "✅ Cargado" if health["model_regression_loaded"]    else "❌ No cargado")
except requests.exceptions.ConnectionError:
    st.warning("API no disponible. El simulador de ROI sigue funcionando.")

st.divider()

# ── Métricas del entrenamiento ─────────────────────────────────────────────────
st.subheader("📋 Métricas de Rendimiento — Modelos ANN")
st.markdown("Resultados obtenidos al finalizar el entrenamiento de ambos modelos con el dataset Telco Customer Churn (7,032 clientes).")

st.markdown("#### Modelo 1 — Clasificación Binaria (Churn Yes/No)")
c1, c2, c3, c4 = st.columns(4)
acc       = c1.number_input("Accuracy",  min_value=0.0, max_value=1.0, value=0.84, step=0.01, format="%.4f")
precision = c2.number_input("Precision", min_value=0.0, max_value=1.0, value=0.67, step=0.01, format="%.4f")
recall    = c3.number_input("Recall",    min_value=0.0, max_value=1.0, value=0.78, step=0.01, format="%.4f")
f1        = c4.number_input("F1-Score",  min_value=0.0, max_value=1.0, value=0.72, step=0.01, format="%.4f")

# Radar con el perfil de métricas del Modelo 1
fig_radar = go.Figure(go.Scatterpolar(
    r=[acc, precision, recall, f1, acc],
    theta=["Accuracy", "Precision", "Recall", "F1-Score", "Accuracy"],
    fill="toself",
    line_color="#2E86AB",
    name="Clasificación",
))
fig_radar.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    title="Perfil de métricas — Modelo de Clasificación",
    height=350,
)
st.plotly_chart(fig_radar, use_container_width=True)

st.markdown("#### Modelo 2 — Regresión / Scoring de Riesgo (0.0 a 1.0)")
c1, c2 = st.columns(2)
mae = c1.number_input("MAE (Error Absoluto Medio)", min_value=0.0, max_value=1.0, value=0.20, step=0.01, format="%.4f")
c2.markdown(f"""
**¿Por qué MAE?**  
El Modelo 2 predice una probabilidad continua entre 0 y 1.
Un MAE de **{mae:.2f}** significa que en promedio la predicción
se desvía ±{mae:.2f} del valor real.
El objetivo del proyecto era MAE ≤ 0.22 — {'✅ cumplido' if mae <= 0.22 else '⚠️ revisar'}.
""")

# Barra comparativa MAE obtenido vs objetivo del proyecto
fig_mae = go.Figure()
fig_mae.add_trace(go.Bar(
    x=["MAE obtenido", "Objetivo máximo"],
    y=[mae, 0.22],
    marker_color=["#2ECC71" if mae <= 0.22 else "#E74C3C", "#E74C3C"],
    text=[f"{mae:.4f}", "0.22"],
    textposition="outside",
))
fig_mae.update_layout(
    title="MAE del Modelo de Regresión vs objetivo",
    yaxis_range=[0, 0.35],
    height=320,
)
st.plotly_chart(fig_mae, use_container_width=True)

st.divider()

# ── Simulador de ROI ───────────────────────────────────────────────────────────
st.subheader("💰 Simulador de ROI — Estrategia de Retención")
st.markdown(
    "Estimación del retorno de inversión de una campaña de retención "
    "dirigida a los clientes identificados como **riesgo Alto o Medio** por el modelo de scoring."
)

c1, c2, c3 = st.columns(3)
n_clientes_riesgo = c1.number_input("Clientes en riesgo (Alto + Medio)", min_value=0, value=1500, step=10)
costo_retencion   = c2.number_input("Costo de retención por cliente (USD)", min_value=0.0, value=15.0, step=1.0)
tasa_exito        = c3.number_input("Tasa de éxito de la campaña (%)", min_value=0.0, max_value=100.0, value=25.0, step=1.0)

c1, c2 = st.columns(2)
arpu            = c1.number_input("Ingreso mensual promedio por cliente — ARPU (USD)", min_value=0.0, value=65.0, step=1.0)
meses_retencion = c2.number_input("Meses de retención esperada", min_value=1, value=12, step=1)

# Cálculo del ROI basado en los parámetros ingresados
inversion_total    = n_clientes_riesgo * costo_retencion
clientes_retenidos = n_clientes_riesgo * (tasa_exito / 100)
ingreso_recuperado = clientes_retenidos * arpu * meses_retencion
roi = ((ingreso_recuperado - inversion_total) / inversion_total * 100) if inversion_total > 0 else 0

st.divider()

c1, c2, c3, c4 = st.columns(4)
c1.metric("💸 Inversión total",              f"${inversion_total:,.2f}")
c2.metric("👥 Clientes retenidos estimados", f"{clientes_retenidos:,.0f}")
c3.metric("📈 Ingreso recuperado",           f"${ingreso_recuperado:,.2f}")
c4.metric("🎯 ROI estimado",                 f"{roi:,.1f}%")

# Comparación visual entre inversión e ingreso recuperado
fig_bar = go.Figure()
fig_bar.add_trace(go.Bar(name="Inversión",          x=["Campaña de retención"], y=[inversion_total],    marker_color="#E74C3C"))
fig_bar.add_trace(go.Bar(name="Ingreso recuperado", x=["Campaña de retención"], y=[ingreso_recuperado], marker_color="#2ECC71"))
fig_bar.update_layout(barmode="group", title="Inversión vs Ingreso recuperado", yaxis_title="USD")
st.plotly_chart(fig_bar, use_container_width=True)

if roi > 0:
    st.success(f"✅ ROI positivo estimado de **{roi:,.1f}%** — la campaña es rentable con los parámetros actuales.")
else:
    st.error(f"⚠️ ROI negativo estimado de **{roi:,.1f}%** — ajustá los parámetros para mejorar la rentabilidad.")

st.divider()

# ── Sensibilidad del ROI ───────────────────────────────────────────────────────
# Muestra cómo varía el ROI según diferentes tasas de éxito posibles
st.subheader("📊 Sensibilidad del ROI según tasa de éxito")
tasas = list(range(5, 105, 5))
rois  = []
for t in tasas:
    retenidos = n_clientes_riesgo * (t / 100)
    ingreso   = retenidos * arpu * meses_retencion
    r = ((ingreso - inversion_total) / inversion_total * 100) if inversion_total > 0 else 0
    rois.append(r)

sens_df = pd.DataFrame({"Tasa de éxito (%)": tasas, "ROI (%)": rois})
fig_sens = px.line(
    sens_df, x="Tasa de éxito (%)", y="ROI (%)", markers=True,
    title="ROI proyectado según tasa de éxito de la campaña",
)
fig_sens.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Punto de equilibrio")
st.plotly_chart(fig_sens, use_container_width=True)
