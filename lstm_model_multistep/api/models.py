from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class PredictionRequest(BaseModel):
    data: List[float]

class PredictionResponse(BaseModel):
    result: List[float]

class MultiStepPredictionRequest(BaseModel):
    data: List[float]
    forecast_steps: int = 1

class MultiStepPredictionResponse(BaseModel):
    result: List[List[float]]  # Shape: (forecast_steps, output_features)
    forecast_steps: int

class ForecastRequest(BaseModel):
    model_name: str
    start_date: str  # ISO format: "2024-01-15"
    horizon_days: int = 3
    features: Optional[List[str]] = None

class ForecastResponse(BaseModel):
    model_name: str
    timestamps: List[str]  # ISO format timestamps
    forecast_power: List[float]  # P = U * I
    forecast_voltage: List[float]  # U
    forecast_current: List[float]  # I
    confidence_intervals: Optional[List[float]] = None
    metadata: dict

class ModelInfoResponse(BaseModel):
    model_name: str
    model_type: str
    training_date: str
    performance_metrics: dict
    input_features: List[str]
    output_features: List[str]