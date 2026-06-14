"""
4_Comparacion.py - Comparación de rendimiento entre el Modelo 1
(Clasificación Binaria) y el Modelo 2 (Regresión / Scoring de Riesgo).
Justificación de la selección del modelo principal.

Proyecto B: Sistema de Predicción de Abandono de Clientes (Churn)
BD-151 Inteligencia Artificial Aplicada
Colegio Universitario de Cartago - 2026
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Comparación de Modelos | Churn", page_icon="⚖️", layout="wide")

st.title("⚖️ Comparación de Modelos")
st.caption("BD-151 Inteligencia Artificial Aplicada — Colegio Universitario de Cartago")
st.markdown("Análisis comparativo entre el **Modelo 1 (Clasificación Binaria)** y el **Modelo 2 (Regresión / Scoring de Riesgo)**, y justificación de la selección del modelo principal.")

st.divider()

# ── Tabla comparativa ──────────────────────────────────────────────────────────
st.subheader("📋 Comparación de Métricas")

tabla = pd.DataFrame({
    "Métrica":                  ["Accuracy", "Precision", "Recall", "F1-Score", "MAE", "Objetivo cumplido", "Uso principal"],
    "Modelo 1 — Clasificación": ["0.84",     "0.67",      "0.78",   "0.72",     "—",   "✅ Accuracy ≥ 0.84", "Decisión binaria: Churn Yes / No"],
    "Modelo 2 — Regresión":     ["—",        "—",         "—",      "—",        "0.20","✅ MAE ≤ 0.22",      "Scoring: probabilidad 0.0 a 1.0"],
})
st.dataframe(tabla.set_index("Métrica"), use_container_width=True)

st.divider()

# ── Gráficas comparativas ──────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Métricas del Modelo 1")
    fig_radar = go.Figure(go.Scatterpolar(
        r=[0.84, 0.67, 0.78, 0.72, 0.84],
        theta=["Accuracy", "Precision", "Recall", "F1-Score", "Accuracy"],
        fill="toself",
        line_color="#2E86AB",
        name="Clasificación",
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Perfil de métricas — Clasificación",
        height=380,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

with col2:
    st.subheader("📉 MAE del Modelo 2 vs Objetivo")
    fig_mae = go.Figure()
    fig_mae.add_trace(go.Bar(
        x=["MAE obtenido", "Objetivo máximo"],
        y=[0.20, 0.22],
        marker_color=["#2ECC71", "#E74C3C"],
        text=["0.20", "0.22"],
        textposition="outside",
    ))
    fig_mae.update_layout(
        title="MAE obtenido vs objetivo del proyecto",
        yaxis_range=[0, 0.35],
        height=380,
    )
    st.plotly_chart(fig_mae, use_container_width=True)

st.divider()

# ── Barras comparativas de métricas compartidas ────────────────────────────────
st.subheader("📊 Comparación visual de métricas en común")

metricas = ["Accuracy", "Precision", "Recall", "F1-Score"]
valores_m1 = [0.84, 0.67, 0.78, 0.72]

fig_bar = go.Figure()
fig_bar.add_trace(go.Bar(
    name="Modelo 1 — Clasificación",
    x=metricas,
    y=valores_m1,
    marker_color="#2E86AB",
    text=[f"{v:.2f}" for v in valores_m1],
    textposition="outside",
))
fig_bar.update_layout(
    title="Métricas de clasificación — Modelo 1",
    yaxis_range=[0, 1.1],
    yaxis_title="Valor",
    height=380,
)
st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# ── Conclusión ─────────────────────────────────────────────────────────────────
st.subheader("🧠 Conclusión — ¿Por qué el Modelo 1 es el modelo principal?")

col1, col2 = st.columns(2)

with col1:
    st.success("""
    ### ✅ Modelo 1 — Clasificación (PRINCIPAL)

    - Responde directamente la pregunta del negocio: ¿este cliente va a abandonar o no?
    - Accuracy de 0.84 — cumple el objetivo definido para el proyecto
    - La salida es binaria (Yes/No), fácil de interpretar y accionar
    - Permite tomar decisiones inmediatas de retención sin ambigüedad
    """)

with col2:
    st.warning("""
    ### 🔄 Modelo 2 — Regresión (COMPLEMENTARIO)

    - No responde Sí/No, sino que estima una probabilidad continua entre 0.0 y 1.0
    - MAE de 0.20 — cumple el objetivo de ≤ 0.22 ✅
    - Su valor está en la **priorización**: permite ordenar a los clientes en riesgo
    - Alimenta la segmentación Bajo / Medio / Alto del dashboard de análisis
    - No reemplaza al Modelo 1, sino que lo complementa con más granularidad
    """)

st.info("""
**📌 Uso conjunto de ambos modelos**

El Modelo 1 responde *¿hay que retener a este cliente?*
El Modelo 2 responde *¿con qué urgencia hay que contactarlo?*

Juntos forman un sistema completo: el primero filtra, el segundo prioriza.
Esto permite que el equipo de retención enfoque sus recursos en los clientes
con mayor probabilidad de abandono primero, maximizando el impacto de la campaña.
""")
