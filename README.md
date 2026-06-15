# Sistema-de-prediccion-de-abandono-de-clientes
# 🤖 Predicción de Churn de Clientes utilizando Redes Neuronales Artificiales

## 📖 Introducción

En este proyecto se desarrolló un sistema basado en Inteligencia Artificial capaz de predecir el abandono de clientes (Customer Churn) en una empresa de telecomunicaciones.

El objetivo principal fue aplicar los conocimientos adquiridos durante el curso de Inteligencia Artificial para resolver un problema real mediante técnicas de análisis de datos y aprendizaje automático.

Para lograrlo, se utilizaron Redes Neuronales Artificiales (ANN), las cuales fueron entrenadas con información histórica de clientes para identificar patrones que permitan anticipar posibles abandonos del servicio.

---

## 🎯 Objetivo General

Desarrollar modelos de Inteligencia Artificial capaces de predecir el abandono de clientes utilizando información histórica de una empresa de telecomunicaciones.

## 📌 Objetivos Específicos

✅ Realizar un análisis exploratorio de los datos.

✅ Aplicar técnicas de limpieza y preprocesamiento.

✅ Construir un modelo de clasificación mediante Redes Neuronales Artificiales.

✅ Desarrollar un modelo de regresión para estimar probabilidades de abandono.

✅ Comparar el desempeño de ambos modelos utilizando diferentes métricas.

---

## 📊 Descripción del Dataset

Para este proyecto se utilizó el conjunto de datos **Telco Customer Churn**, ampliamente utilizado en problemas de clasificación y análisis de clientes.

El dataset contiene información relacionada con:

📍 Género del cliente.

📍 Antigüedad dentro de la empresa.

📍 Tipo de contrato.

📍 Servicios contratados.

📍 Método de pago.

📍 Cargos mensuales y totales.

📍 Estado de abandono del cliente.

La variable principal del estudio es **Churn**, que indica si un cliente abandonó o no el servicio.

---

## 🛠️ Desarrollo del Proyecto

El proyecto fue desarrollado siguiendo una serie de etapas que permitieron preparar los datos y entrenar los modelos de manera adecuada.

### 🔍 1. Análisis Exploratorio de Datos (EDA)

Durante esta fase se realizó una exploración inicial del conjunto de datos con el objetivo de comprender mejor la información disponible.

Algunas actividades realizadas fueron:

- 📈 Análisis de distribuciones.
- 🔎 Identificación de valores faltantes.
- 📊 Visualización de variables relevantes.
- 🔗 Estudio de correlaciones.
- 🧹 Revisión de posibles inconsistencias.

Esta etapa permitió conocer mejor el comportamiento de los clientes antes de comenzar con el modelado.

---

### ⚙️ 2. Preprocesamiento de Datos

Una vez completado el análisis exploratorio, se prepararon los datos para el entrenamiento de los modelos.

Entre las tareas realizadas se encuentran:

- 🧹 Limpieza de datos.
- 🔄 Transformación de variables categóricas.
- 🏷️ Codificación de atributos.
- 📏 Escalado de variables numéricas.
- ⚖️ Balanceo de clases utilizando SMOTE.

Gracias a este proceso fue posible obtener un conjunto de datos más adecuado para el aprendizaje de los modelos.

---

### 🧠 3. Modelo de Clasificación

Se desarrolló una Red Neuronal Artificial cuyo objetivo es determinar si un cliente abandonará o no la empresa.

El modelo fue entrenado utilizando los datos previamente procesados y evaluado mediante métricas de clasificación.

Este enfoque permite identificar clientes con alto riesgo de abandono para que la empresa pueda tomar medidas preventivas.

---

### 📉 4. Modelo de Regresión

Además del modelo de clasificación, se implementó una Red Neuronal orientada a regresión.

Su función consiste en estimar una probabilidad de abandono para cada cliente, proporcionando una visión más detallada del nivel de riesgo asociado.

Este tipo de predicción puede ser útil para priorizar estrategias de retención de clientes.

---

### 📋 5. Comparación de Modelos

Finalmente, se compararon los resultados obtenidos por ambos modelos.

Esta comparación permitió analizar las ventajas y limitaciones de cada enfoque, así como identificar cuál ofrece mejores resultados para el problema planteado.

---

## 💻 Tecnologías Utilizadas

Durante el desarrollo del proyecto se utilizaron las siguientes herramientas:

🐍 Python

📊 Pandas

🔢 NumPy

📈 Matplotlib

🎨 Seaborn

🤖 Scikit-Learn

🧠 TensorFlow

⚡ Keras

⚖️ Imbalanced-Learn (SMOTE)

📓 Jupyter Notebook

---

## 📂 Estructura del Proyecto

```text
📁 Proyecto_Churn
│
├── 📊 01_EDA.ipynb
├── ⚙️ 02_Preprocesamiento.ipynb
├── 🧠 Modelo_Clasificacion.ipynb
├── 📉 Modelo_Regresion.ipynb
├── 📋 Comparacion_Modelos.ipynb
└── 📄 README.md
```

---

## 🚀 Aprendizajes Obtenidos

A través de este proyecto fue posible reforzar conocimientos relacionados con:

- 📊 Análisis exploratorio de datos.
- 🧹 Limpieza y transformación de información.
- 🤖 Machine Learning.
- 🧠 Redes Neuronales Artificiales.
- 📈 Evaluación de modelos predictivos.
- 💻 Desarrollo de soluciones utilizando Python.

---

## 🏆 Conclusiones

Este proyecto permitió aplicar de forma práctica los conceptos aprendidos durante el curso de Inteligencia Artificial.

Los resultados obtenidos demuestran que las Redes Neuronales Artificiales pueden ser utilizadas para identificar clientes con riesgo de abandono y apoyar procesos de toma de decisiones dentro de una organización.

Además, el trabajo permitió fortalecer habilidades en análisis de datos, preparación de información y construcción de modelos predictivos.

---

## 👨‍🎓 Autor

Proyecto desarrollado para el curso:

**BD-151 Inteligencia Artificial Aplicada**

🏫 Colegio Universitario de Cartago (CUC)

📅 2026cimientos adquiridos durante el curso.
