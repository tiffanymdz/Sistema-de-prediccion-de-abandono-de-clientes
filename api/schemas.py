"""
Esquemas Pydantic para la API de Predicción de Abandono de Clientes (Churn).

Estos modelos validan los datos de entrada que llegan a los endpoints
/predict/churn y /predict/risk_score, y definen el formato de las
respuestas.
"""

from pydantic import BaseModel, Field
from typing import Literal


class ClienteInput(BaseModel):
    """
    Datos de un cliente, en el mismo formato que las columnas
    del dataset 'customer_clean.csv' (después del preprocesamiento).

    Notas sobre codificación:
    - Variables binarias (gender, Partner, Dependents, etc.): 0 = No, 1 = Sí
      (gender: 0 = Female, 1 = Male, según LabelEncoder original)
    - Contract_*, PaymentMethod_*, InternetService_*: variables one-hot,
      solo UNA de cada grupo debe ser 1 y las demás 0.
    """

    gender: int = Field(..., ge=0, le=1, description="0 = Female, 1 = Male")
    SeniorCitizen: int = Field(..., ge=0, le=1, description="1 si es adulto mayor")
    Partner: int = Field(..., ge=0, le=1, description="1 si tiene pareja")
    Dependents: int = Field(..., ge=0, le=1, description="1 si tiene dependientes")
    tenure: int = Field(..., ge=0, le=100, description="Meses de antigüedad como cliente")
    PhoneService: int = Field(..., ge=0, le=1)
    MultipleLines: int = Field(..., ge=0, le=1)
    OnlineSecurity: int = Field(..., ge=0, le=1)
    OnlineBackup: int = Field(..., ge=0, le=1)
    DeviceProtection: int = Field(..., ge=0, le=1)
    TechSupport: int = Field(..., ge=0, le=1)
    StreamingTV: int = Field(..., ge=0, le=1)
    StreamingMovies: int = Field(..., ge=0, le=1)
    PaperlessBilling: int = Field(..., ge=0, le=1)
    MonthlyCharges: float = Field(..., ge=0, description="Cargo mensual en USD")
    TotalCharges: float = Field(..., ge=0, description="Cargo total acumulado en USD")

    # One-hot: Contract (Month-to-month / One year / Two year)
    Contract_Month_to_month: int = Field(0, ge=0, le=1, alias="Contract_Month-to-month")
    Contract_One_year: int = Field(0, ge=0, le=1, alias="Contract_One year")
    Contract_Two_year: int = Field(0, ge=0, le=1, alias="Contract_Two year")

    # One-hot: PaymentMethod
    PaymentMethod_Bank_transfer: int = Field(0, ge=0, le=1, alias="PaymentMethod_Bank transfer (automatic)")
    PaymentMethod_Credit_card: int = Field(0, ge=0, le=1, alias="PaymentMethod_Credit card (automatic)")
    PaymentMethod_Electronic_check: int = Field(0, ge=0, le=1, alias="PaymentMethod_Electronic check")
    PaymentMethod_Mailed_check: int = Field(0, ge=0, le=1, alias="PaymentMethod_Mailed check")

    # One-hot: InternetService
    InternetService_DSL: int = Field(0, ge=0, le=1)
    InternetService_Fiber_optic: int = Field(0, ge=0, le=1, alias="InternetService_Fiber optic")
    InternetService_No: int = Field(0, ge=0, le=1)

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "gender": 1,
                "SeniorCitizen": 0,
                "Partner": 1,
                "Dependents": 0,
                "tenure": 12,
                "PhoneService": 1,
                "MultipleLines": 0,
                "OnlineSecurity": 0,
                "OnlineBackup": 1,
                "DeviceProtection": 0,
                "TechSupport": 0,
                "StreamingTV": 1,
                "StreamingMovies": 1,
                "PaperlessBilling": 1,
                "MonthlyCharges": 85.5,
                "TotalCharges": 1026.0,
                "Contract_Month-to-month": 1,
                "Contract_One year": 0,
                "Contract_Two year": 0,
                "PaymentMethod_Bank transfer (automatic)": 0,
                "PaymentMethod_Credit card (automatic)": 0,
                "PaymentMethod_Electronic check": 1,
                "PaymentMethod_Mailed check": 0,
                "InternetService_DSL": 0,
                "InternetService_Fiber optic": 1,
                "InternetService_No": 0,
            }
        }


class ChurnPredictionOutput(BaseModel):
    churn_prediction: Literal["Yes", "No"]
    churn_probability: float = Field(..., description="Probabilidad de churn (0-1) según el modelo de clasificación")
    confidence: float = Field(..., description="Confianza de la predicción (0-1)")


class RiskScoreOutput(BaseModel):
    risk_score: float = Field(..., description="Score de riesgo de churn (0.0 a 1.0)")
    risk_level: Literal["Bajo", "Medio", "Alto"]
    recommended_action: str


class HealthOutput(BaseModel):
    status: str
    model_classification_loaded: bool
    model_regression_loaded: bool
