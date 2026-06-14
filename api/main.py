"""
API REST - Sistema de Predicción de Abandono de Clientes (Churn)
Colegio Universitario de Cartago - BD-151

Endpoints:
- GET  /                   -> info general
- GET  /health             -> estado de la API y modelos cargados
- POST /predict/churn      -> Modelo 1: clasificación binaria (Churn Yes/No)
- POST /predict/risk_score -> Modelo 2: regresión / score de riesgo (0.0 - 1.0)

Ejecutar:
    uvicorn api.main:app --reload
Documentación automática:
    http://127.0.0.1:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import ClienteInput, ChurnPredictionOutput, RiskScoreOutput, HealthOutput
from api.predict import model_service

app = FastAPI(
    title="API - Predicción de Abandono de Clientes (Churn)",
    description=(
        "API REST para predecir el abandono (churn) de clientes de "
        "telecomunicaciones y calcular su nivel de riesgo, usando dos "
        "modelos de redes neuronales (ANN) entrenados con TensorFlow/Keras."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
def root():
    return {
        "mensaje": "API de Predicción de Abandono de Clientes (Churn)",
        "endpoints": ["/health", "/predict/churn", "/predict/risk_score", "/docs"],
    }


@app.get("/health", response_model=HealthOutput, tags=["General"])
def health():
    return HealthOutput(
        status="ok" if model_service.is_ready else "modelos no cargados",
        model_classification_loaded=model_service.model_class is not None,
        model_regression_loaded=model_service.model_reg is not None,
    )


@app.post("/predict/churn", response_model=ChurnPredictionOutput, tags=["Predicciones"])
def predict_churn(cliente: ClienteInput):
    """
    Modelo 1 - Clasificación binaria.
    Predice si el cliente abandonará el servicio (Yes/No) junto con
    la probabilidad y confianza de la predicción.
    """
    if not model_service.is_ready:
        raise HTTPException(status_code=503, detail="Los modelos no están cargados. Verifique la carpeta 'models/'.")

    try:
        payload = cliente.dict(by_alias=True)
        df = model_service.input_to_dataframe(payload)
        prediction, prob, confidence = model_service.predict_churn(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error procesando la solicitud: {e}")

    return ChurnPredictionOutput(
        churn_prediction=prediction,
        churn_probability=round(prob, 4),
        confidence=round(confidence, 4),
    )


@app.post("/predict/risk_score", response_model=RiskScoreOutput, tags=["Predicciones"])
def predict_risk_score(cliente: ClienteInput):
    """
    Modelo 2 - Regresión / scoring de riesgo.
    Calcula la probabilidad de churn (0.0 a 1.0) y clasifica al cliente
    en un nivel de riesgo (Bajo/Medio/Alto) con una recomendación de acción.
    """
    if not model_service.is_ready:
        raise HTTPException(status_code=503, detail="Los modelos no están cargados. Verifique la carpeta 'models/'.")

    try:
        payload = cliente.dict(by_alias=True)
        df = model_service.input_to_dataframe(payload)
        score, level, action = model_service.predict_risk_score(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error procesando la solicitud: {e}")

    return RiskScoreOutput(
        risk_score=round(score, 4),
        risk_level=level,
        recommended_action=action,
    )
