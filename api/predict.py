"""
Lógica de inferencia: carga de modelos .keras, scaler y columnas,
y funciones para transformar la entrada y producir predicciones.
"""

import os
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Rutas base (relativas a la carpeta del proyecto)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_CLASS_PATH = os.path.join(MODELS_DIR, "model_churn_class.keras")
MODEL_REG_PATH = os.path.join(MODELS_DIR, "model_churn_reg.keras")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(MODELS_DIR, "feature_columns.pkl")

# Mapeo de alias (nombres "amigables" usados en el JSON) -> nombres reales de columna
ALIAS_TO_COLUMN = {
    "Contract_Month-to-month": "Contract_Month-to-month",
    "Contract_One year": "Contract_One year",
    "Contract_Two year": "Contract_Two year",
    "PaymentMethod_Bank transfer (automatic)": "PaymentMethod_Bank transfer (automatic)",
    "PaymentMethod_Credit card (automatic)": "PaymentMethod_Credit card (automatic)",
    "PaymentMethod_Electronic check": "PaymentMethod_Electronic check",
    "PaymentMethod_Mailed check": "PaymentMethod_Mailed check",
    "InternetService_Fiber optic": "InternetService_Fiber optic",
}


class ModelService:
    """Carga única (singleton) de modelos y artefactos de preprocesamiento."""

    def __init__(self):
        self.model_class = None
        self.model_reg = None
        self.scaler = None
        self.feature_columns = None
        self._load()

    def _load(self):
        if os.path.exists(MODEL_CLASS_PATH):
            self.model_class = load_model(MODEL_CLASS_PATH)
        if os.path.exists(MODEL_REG_PATH):
            self.model_reg = load_model(MODEL_REG_PATH)
        if os.path.exists(SCALER_PATH):
            with open(SCALER_PATH, "rb") as f:
                self.scaler = pickle.load(f)
        if os.path.exists(FEATURES_PATH):
            with open(FEATURES_PATH, "rb") as f:
                self.feature_columns = pickle.load(f)

    @property
    def is_ready(self) -> bool:
        return all([
            self.model_class is not None,
            self.model_reg is not None,
            self.scaler is not None,
            self.feature_columns is not None,
        ])

    def input_to_dataframe(self, payload: dict) -> pd.DataFrame:
        """
        Convierte el dict recibido (usando alias con espacios/guiones)
        en un DataFrame con el orden exacto de columnas usado al entrenar.
        """
        row = {}
        for col in self.feature_columns:
            if col in payload:
                row[col] = payload[col]
            else:
                # Buscar por alias (claves con espacios/guiones)
                found = None
                for alias, real_col in ALIAS_TO_COLUMN.items():
                    if real_col == col and alias in payload:
                        found = payload[alias]
                        break
                row[col] = found if found is not None else 0
        return pd.DataFrame([row], columns=self.feature_columns)

    def scale(self, df: pd.DataFrame) -> np.ndarray:
        return self.scaler.transform(df)

    def predict_churn(self, df: pd.DataFrame):
        X_scaled = self.scale(df)
        prob = float(self.model_class.predict(X_scaled, verbose=0).flatten()[0])
        prediction = "Yes" if prob >= 0.5 else "No"
        confidence = prob if prediction == "Yes" else (1 - prob)
        return prediction, prob, confidence

    def predict_risk_score(self, df: pd.DataFrame):
        X_scaled = self.scale(df)
        score = float(self.model_reg.predict(X_scaled, verbose=0).flatten()[0])
        score = max(0.0, min(1.0, score))
        level = segmentar_riesgo(score)
        action = recomendar_accion(level)
        return score, level, action


def segmentar_riesgo(prob: float) -> str:
    if prob < 0.33:
        return "Bajo"
    elif prob < 0.66:
        return "Medio"
    else:
        return "Alto"


def recomendar_accion(level: str) -> str:
    acciones = {
        "Bajo": "Cliente estable. Mantener calidad de servicio actual, sin acción urgente.",
        "Medio": "Ofrecer beneficios de fidelización (descuento, upgrade de plan o soporte proactivo).",
        "Alto": "Contacto prioritario del equipo de retención: oferta personalizada y revisión de contrato.",
    }
    return acciones.get(level, "Sin recomendación disponible.")


# Instancia global reutilizada por la API
model_service = ModelService()
