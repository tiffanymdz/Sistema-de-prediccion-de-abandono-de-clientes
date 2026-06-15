# Sistema-de-prediccion-de-abandono-de-clientes
Desarrollar un sistema que prediga qué clientes tienen mayor probabilidad de abandonar un servicio de telecomunicaciones.
Introducción

En este proyecto se desarrolló un modelo de Inteligencia Artificial capaz de predecir el abandono de clientes (Customer Churn) en una empresa de telecomunicaciones. El objetivo principal fue aplicar los conocimientos adquiridos en el curso de Inteligencia Artificial para resolver un problema real relacionado con la retención de clientes.

El trabajo se realizó utilizando Python y diferentes bibliotecas de análisis de datos y aprendizaje automático. Además, se implementaron Redes Neuronales Artificiales (ANN) para construir modelos capaces de identificar patrones en la información de los clientes y realizar predicciones.

Objetivo General

Desarrollar e implementar modelos de Inteligencia Artificial que permitan predecir el abandono de clientes utilizando información histórica de una empresa de telecomunicaciones.

Objetivos Específicos
Realizar un análisis exploratorio de los datos para comprender las características del conjunto de datos.
Aplicar técnicas de limpieza y preprocesamiento para preparar los datos antes del entrenamiento.
Construir un modelo de clasificación basado en Redes Neuronales Artificiales.
Construir un modelo de regresión para estimar la probabilidad de abandono.
Comparar el desempeño de ambos modelos utilizando métricas de evaluación adecuadas.
Descripción del Dataset

Para este proyecto se utilizó el conjunto de datos Telco Customer Churn, el cual contiene información de clientes de una empresa de telecomunicaciones.

Entre las variables incluidas se encuentran:

Género del cliente.
Antigüedad en la empresa.
Tipo de contrato.
Servicios contratados.
Método de pago.
Cargos mensuales.
Cargos totales.
Estado de abandono del cliente.

La variable más importante del conjunto de datos es Churn, ya que indica si un cliente abandonó o no el servicio.

Desarrollo del Proyecto

El proyecto se dividió en varias etapas con el fin de seguir una metodología ordenada para el análisis y modelado de los datos.

1. Análisis Exploratorio de Datos (EDA)

En esta etapa se realizó una revisión general del conjunto de datos para identificar posibles problemas y comprender mejor la información disponible.

Entre las actividades realizadas se encuentran:

Inspección de tipos de datos.
Identificación de valores faltantes.
Análisis de distribuciones.
Visualización de variables relevantes.
Estudio de correlaciones entre características.

Este análisis permitió conocer mejor el comportamiento de los clientes y detectar aspectos importantes para las siguientes etapas del proyecto.

2. Preprocesamiento de Datos

Una vez realizado el análisis exploratorio, se procedió a preparar los datos para el entrenamiento de los modelos.

Las tareas realizadas incluyeron:

Limpieza de registros inconsistentes.
Transformación de variables categóricas a valores numéricos.
Aplicación de técnicas de codificación.
Escalado de variables numéricas.
Balanceo de clases mediante SMOTE.

El preprocesamiento fue una etapa fundamental para mejorar la calidad de los datos y facilitar el aprendizaje de los modelos.

3. Modelo de Clasificación

Posteriormente se desarrolló un modelo de clasificación utilizando Redes Neuronales Artificiales.

Este modelo tiene como objetivo determinar si un cliente tiene probabilidades de abandonar la empresa o permanecer como cliente activo.

Durante el entrenamiento se ajustaron diferentes parámetros para mejorar el rendimiento del modelo y obtener resultados más precisos.

Las métricas de evaluación utilizadas permitieron medir la capacidad del modelo para identificar correctamente los casos de abandono.

4. Modelo de Regresión

Además del modelo de clasificación, se desarrolló un modelo de regresión utilizando una arquitectura similar de red neuronal.

El propósito de este modelo fue estimar una probabilidad de abandono para cada cliente, proporcionando una medida más detallada del riesgo asociado a cada caso.

Este enfoque puede resultar útil para que las empresas establezcan prioridades en sus estrategias de retención.

5. Comparación de Resultados

Finalmente, se compararon los resultados obtenidos por ambos modelos con el fin de evaluar sus fortalezas y limitaciones.

La comparación permitió analizar cuál de los enfoques ofrece mejores resultados para el problema planteado y cómo pueden complementarse dentro de un sistema de apoyo para la toma de decisiones.

Herramientas Utilizadas

Durante el desarrollo del proyecto se utilizaron las siguientes tecnologías:

Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-Learn
TensorFlow
Keras
Imbalanced-Learn (SMOTE)
Jupyter Notebook
Estructura del Proyecto
01_EDA.ipynb                 -> Análisis exploratorio de datos
02_Preprocesamiento.ipynb    -> Limpieza y preparación de datos
Modelo_Clasificacion.ipynb   -> Red neuronal para clasificación
Modelo_Regresion.ipynb       -> Red neuronal para regresión
Comparacion_Modelos.ipynb    -> Comparación de resultados
README.md                    -> Documentación del proyecto
Conclusiones

La realización de este proyecto permitió aplicar de forma práctica conceptos relacionados con análisis de datos, aprendizaje automático y redes neuronales artificiales.

Los resultados obtenidos demuestran que es posible utilizar modelos de Inteligencia Artificial para identificar clientes con riesgo de abandono y apoyar la toma de decisiones dentro de una organización.

Además, el proyecto permitió reforzar habilidades en el manejo de datos, preprocesamiento, construcción de modelos y evaluación de resultados, consolidando los conocimientos adquiridos durante el curso.
