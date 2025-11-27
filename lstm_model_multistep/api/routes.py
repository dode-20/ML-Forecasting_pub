from fastapi import APIRouter, HTTPException
from .models import PredictionRequest, PredictionResponse, MultiStepPredictionRequest, MultiStepPredictionResponse, ForecastRequest, ForecastResponse, ModelInfoResponse
from service.src.model import LSTMModel
import torch
from typing import List
import json
import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from pathlib import Path

router = APIRouter()

@router.get("/health")
def health_check():
    return {"status": "ok"}

@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    # Initialisiere das Modell mit den korrekten Parametern
    input_data: List[List[float]] = [request.data]  # Wrap in list for batch dimension
    model = LSTMModel(
        input_size=len(input_data[0]) if input_data else 1,  # Anzahl der Features
        hidden_size=64,
        num_layers=2,
        output_size=1,  # Ein Ausgabewert pro Zeitschritt
        dropout=0.2
    )
    # Load model from the new structure - for now, load the most recent model
    # Check if running in container (results mounted as /app/results) or local development
    if Path("/app/results").exists():
        models_dir = "/app/results/trained_models/lstm"
    else:
        models_dir = "results/trained_models/lstm"
    if os.path.exists(models_dir):
        # Find the most recent model directory
        model_dirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d)) and d.endswith('_lstm')]
        if model_dirs:
            # Sort by timestamp (newest first) and take the first one
            model_dirs.sort(reverse=True)
            latest_model_dir = model_dirs[0]
            model_name = latest_model_dir.replace('_lstm', '')
            model_file = os.path.join(models_dir, latest_model_dir, f"{model_name}_lstm.pth")
            if os.path.exists(model_file):
                model.load_state_dict(torch.load(model_file, map_location=torch.device("cpu")))
            else:
                raise HTTPException(status_code=404, detail="No trained model found")
        else:
            raise HTTPException(status_code=404, detail="No trained models found")
    else:
        raise HTTPException(status_code=404, detail="Models directory not found")
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(input_data).float()  # (batch_size, seq_len, features)
        prediction = model(input_tensor)
    return PredictionResponse(result=prediction.squeeze().tolist())

@router.post("/predict-multistep", response_model=MultiStepPredictionResponse)
def predict_multistep(request: MultiStepPredictionRequest):
    """Multi-step ahead prediction endpoint"""
    try:
        # Load the most recent model
        if Path("/app/results").exists():
            models_dir = "/app/results/trained_models/lstm"
        else:
            models_dir = "results/trained_models/lstm"
            
        model_dirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d)) and d.endswith('_lstm')]
        if not model_dirs:
            raise HTTPException(status_code=404, detail="No trained models found")
            
        model_dirs.sort(reverse=True)
        latest_model_dir = model_dirs[0]
        model_name = latest_model_dir.replace('_lstm', '')
        
        # Load model config to get parameters
        config_path = os.path.join(models_dir, latest_model_dir, f"model_config_{model_name}_lstm.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Get model parameters
        training_settings = config.get("training_settings", {})
        input_size = len(training_settings.get("features", []))
        output_size = len(training_settings.get("output", []))
        hidden_size = config.get("hidden_size", 128)
        num_layers = config.get("num_layers", 3)
        dropout = config.get("dropout", 0.3)
        forecast_steps = request.forecast_steps
        
        # Initialize model
        model = LSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            dropout=dropout,
            forecast_steps=forecast_steps
        )
        
        # Load model weights
        model_path = os.path.join(models_dir, latest_model_dir, f"{model_name}_lstm.pth")
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        model.eval()
        
        # Prepare input data
        input_data = torch.tensor([request.data]).float()  # (batch_size, seq_len, features)
        
        # Make prediction
        with torch.no_grad():
            prediction = model(input_data)  # Shape: (batch_size, forecast_steps, output_size)
        
        # Convert to list format
        result = prediction.squeeze(0).tolist()  # Shape: (forecast_steps, output_size)
        
        return MultiStepPredictionResponse(
            result=result,
            forecast_steps=forecast_steps
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/forecast", response_model=ForecastResponse)
def get_forecast(request: ForecastRequest):
    """Generates a forecast for a given time period"""
    try:
        # Parse start date
        start_date = datetime.fromisoformat(request.start_date)
        horizon_days = request.horizon_days

        # Generate timestamps for the forecast
        timestamps = []
        for day in range(horizon_days):
            for hour in range(24):
                for minute in range(0, 60, 15):  # 15-Minuten-Intervalle
                    dt = start_date + timedelta(days=day, hours=hour, minutes=minute)
                    timestamps.append(dt.isoformat())

        # Load model and preprocessor
        # (We use the latest model with matching config and preprocessor params)
        if Path("/app/results").exists():
            models_dir = "/app/results/trained_models/lstm"
        else:
            models_dir = "results/trained_models/lstm"
        model_dirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d)) and d.endswith('_lstm')]
        if not model_dirs:
            raise HTTPException(status_code=404, detail="No trained models found")
        model_dirs.sort(reverse=True)
        latest_model_dir = model_dirs[0]
        model_name = latest_model_dir.replace('_lstm', '')
        model_path = os.path.join(models_dir, latest_model_dir, f"{model_name}_lstm.pth")
        config_path = os.path.join(models_dir, latest_model_dir, f"model_config_{model_name}_lstm.json")
        # Extract timestamp from model folder name (e.g. 20250702_1256_lstm -> 20250702_1256)
        timestamp = latest_model_dir.split('_lstm')[0]
        preprocessor_path = os.path.join(models_dir, latest_model_dir, f"preprocessor_{timestamp}_lstm.json")

        # Load model config
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Combine time features and sensor features (as for training)
        sensor_features = config["training_settings"]["features"]
        time_features = config.get("time_features", ["day_of_year", "month", "weekday", "hour", "minute"])
        features = time_features + sensor_features  # Time Features VOR Sensor Features
        

        
        # Validate that we have features
        if not features:
            raise HTTPException(status_code=500, detail="No features found in model configuration")
        
        sequence_length = config.get("sequence_length", 864)
        output_features = config["training_settings"]["output"]

        # Load preprocessor
        from service.src.preprocess import DataPreprocessor
        import numpy as np
        import torch
        preprocessor = DataPreprocessor(features=features, output_features=output_features)
        # Load scaler parameters (if available)
        if os.path.exists(preprocessor_path):
            with open(preprocessor_path, "r") as f:
                scaler_params = json.load(f)
            # Set scaler parameters
            scaler_hyperparams = ["clip", "copy", "feature_range"]
            scaler_fitparams = ["min_", "scale_", "data_min_", "data_max_", "data_range_", "n_samples_seen_"]
            for f_name, params in scaler_params.get("feature_scalers", {}).items():
                if f_name in preprocessor.feature_scalers:
                    scaler = preprocessor.feature_scalers[f_name]
                    # Set only hyperparameters via set_params
                    hyper = {k: v for k, v in params.items() if k in scaler_hyperparams}
                    scaler.set_params(**hyper)
                    # Set learned attributes directly
                    for k in scaler_fitparams:
                        if k in params:
                            val = params[k]
                            if k == "n_samples_seen_":
                                setattr(scaler, k, int(val))
                            else:
                                setattr(scaler, k, np.array(val, dtype=np.float64))
            for f_name, params in scaler_params.get("output_scalers", {}).items():
                if f_name in preprocessor.output_scalers:
                    scaler = preprocessor.output_scalers[f_name]
                    hyper = {k: v for k, v in params.items() if k in scaler_hyperparams}
                    scaler.set_params(**hyper)
                    for k in scaler_fitparams:
                        if k in params:
                            val = params[k]
                            if k == "n_samples_seen_":
                                setattr(scaler, k, int(val))
                            else:
                                setattr(scaler, k, np.array(val, dtype=np.float64))

        # Load model
        from service.src.model import LSTMModel
        model = LSTMModel(
            input_size=config["input_size"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            output_size=config["output_size"],
            dropout=config["dropout"]
        )
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        model.eval()

        # Generate input features for the desired period (currently: dummy, since no real input data is available)
        # In real application: prepare input data for the period!
        try:
            input_data_path = "/app/results/forecast_data/aggrData/latest_forecast_input_aggr.csv"
            if not os.path.exists(input_data_path):
                raise FileNotFoundError(f"Input data for forecast not found: {input_data_path}")
            input_df = pd.read_csv(input_data_path)
            
            missing = [f for f in features if f not in input_df.columns]
            if missing:
                raise HTTPException(status_code=500, detail=f"Missing features in input data: {missing}")
            # Ensure only the required features are used
            input_df = input_df[features]
            if len(input_df) < sequence_length:
                raise HTTPException(status_code=500, detail=f"Not enough input data for the required sequence length: {sequence_length}")
            input_df = input_df.tail(sequence_length).reset_index(drop=True)
            X, _ = preprocessor.transform(input_df)
            input_tensor = X.unsqueeze(0)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Could not load or preprocess input data for forecast: {str(e)}")
        with torch.no_grad():
            y_pred = model(input_tensor).squeeze(0).numpy()  # (num_points, n_outputs)
        # Fix: always ensure y_pred is 2D
        if y_pred.ndim == 1:
            y_pred = y_pred[:, np.newaxis]
        # Inverse transformation
        u_i = preprocessor.inverse_transform_output(torch.tensor(y_pred))  # shape: (num_points, n_outputs)
        voltage = u_i[:, 0].tolist()
        current = u_i[:, 1].tolist()
        power = (u_i[:, 0] * u_i[:, 1]).tolist()

        metadata = {
            "model_name": request.model_name,
            "forecast_horizon_days": horizon_days,
            "start_date": request.start_date,
            "data_points": len(power),
            "generated_at": datetime.now().isoformat()
        }

        return ForecastResponse(
            model_name=request.model_name,
            timestamps=timestamps,
            forecast_power=power,
            forecast_voltage=voltage,
            forecast_current=current,
            confidence_intervals=None,
            metadata=metadata
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast generation failed: {str(e)}")

@router.get("/models", response_model=List[ModelInfoResponse])
def get_available_models():
    """Gibt eine Liste aller verfügbaren Modelle zurück"""
    try:
        # Check if running in container (results mounted as /app/results) or local development
        if Path("/app/results").exists():
            models_dir = "/app/results/trained_models/lstm"
        else:
            models_dir = "results/trained_models/lstm"
        available_models = []
        
        if os.path.exists(models_dir):
            # Look for model directories (YYYYMMDD_HHMM_lstm/)
            for model_dir in os.listdir(models_dir):
                if os.path.isdir(os.path.join(models_dir, model_dir)) and model_dir.endswith('_lstm'):
                    model_name = model_dir.replace('_lstm', '')
                    model_dir_path = os.path.join(models_dir, model_dir)
                    
                    # Load model information from config file
                    config_file = os.path.join(model_dir_path, f"model_config_{model_name}_lstm.json")
                    settings = {}
                    if os.path.exists(config_file):
                        with open(config_file, 'r') as f:
                            settings = json.load(f)
                    
                    # Load performance metrics from history file
                    history_file = os.path.join(model_dir_path, f"training_history_{model_name}_lstm.json")
                    performance_metrics = {}
                    if os.path.exists(history_file):
                        with open(history_file, 'r') as f:
                            history = json.load(f)
                            if 'train_loss' in history and 'val_loss' in history:
                                performance_metrics = {
                                    "final_train_loss": history['train_loss'][-1] if history['train_loss'] else None,
                                    "final_val_loss": history['val_loss'][-1] if history['val_loss'] else None,
                                    "epochs_trained": len(history.get('train_loss', []))
                                }
                    
                    model_info = ModelInfoResponse(
                        model_name=model_name,
                        model_type=settings.get('model_type', 'LSTM'),
                        training_date=settings.get('training_date', 'Unknown'),
                        performance_metrics=performance_metrics,
                        input_features=settings.get('training_settings', {}).get('features', []),
                        output_features=settings.get('training_settings', {}).get('output', [])
                    )
                    available_models.append(model_info)
        
        return available_models
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model information: {str(e)}")

@router.get("/models/{model_name}", response_model=ModelInfoResponse)
def get_model_info(model_name: str):
    """Gibt detaillierte Informationen zu einem spezifischen Modell zurück"""
    try:
        # Check if running in container (results mounted as /app/results) or local development
        if Path("/app/results").exists():
            models_dir = "/app/results/trained_models/lstm"
        else:
            models_dir = "results/trained_models/lstm"
        model_dir_path = os.path.join(models_dir, f"{model_name}_lstm")
        model_file = os.path.join(model_dir_path, f"{model_name}_lstm.pth")
        
        if not os.path.exists(model_file):
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
        
        # Load model information
        config_file = os.path.join(model_dir_path, f"model_config_{model_name}_lstm.json")
        settings = {}
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                settings = json.load(f)
        
        # Load performance metrics
        history_file = os.path.join(model_dir_path, f"training_history_{model_name}_lstm.json")
        performance_metrics = {}
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                history = json.load(f)
                if 'train_loss' in history and 'val_loss' in history:
                    performance_metrics = {
                        "final_train_loss": history['train_loss'][-1] if history['train_loss'] else None,
                        "final_val_loss": history['val_loss'][-1] if history['val_loss'] else None,
                        "epochs_trained": len(history.get('train_loss', [])),
                        "loss_history": history
                    }
        
        return ModelInfoResponse(
            model_name=model_name,
            model_type=settings.get('model_type', 'LSTM'),
            training_date=settings.get('training_date', 'Unknown'),
            performance_metrics=performance_metrics,
            input_features=settings.get('training_settings', {}).get('features', []),
            output_features=settings.get('training_settings', {}).get('output', [])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model information: {str(e)}")