from fastapi import FastAPI, Request
from api.routes import router
import httpx
import json
from pathlib import Path
import pandas as pd
from service.src.train import LSTMTrainer
from typing import Dict, Any, Optional, List
from sse_starlette.sse import EventSourceResponse
import asyncio
from pydantic import BaseModel
import sys
import numpy as np
import time
from datetime import datetime

app = FastAPI()
app.include_router(router)

# Global variable for training status
training_status: Dict[str, Dict[str, Any]] = {}
data_preparation_status: Dict[str, Dict[str, Any]] = {}

class TrainingSettings(BaseModel):
    model_name: str
    model_type: str
    module_type: str
    use_all_modules: bool
    selected_modules: List[str]
    features: List[str]
    output: List[str]
    date_selection: Dict[str, Any]
    use_validation_set: str
    epochs: int
    batch_size: int
    learning_rate: float
    shuffle: bool
    loss_function: str

@app.post("/get-training-data")
async def get_training_data(request: Request):
    settings = await request.json()
    model_name = settings.get("model_name")
    print(f"Received training data request for model: {model_name}")
    data_preparation_status[model_name] = {"status": "preparing"}

    async def prepare_data():
        try:
            print(f"[DataPrep] Sending request to InfluxDB for model: {model_name}")
            async with httpx.AsyncClient(timeout=600.0) as client:
                response = await client.post(
                    "http://api_influxdata:8009/query/training_data",
                    json=settings,
                    timeout=600.0
                )
                print(f"[DataPrep] Received HTTP status: {response.status_code}")
                influx_response = response.json()
                print(f"[DataPrep] Full InfluxDB response: {influx_response}")
                if influx_response.get("status") == "success":
                    if influx_response.get("data"):
                        print(f"[DataPrep] InfluxDB returned data with {len(influx_response['data'])} records.")
                    else:
                        print("[DataPrep][Warning] InfluxDB response has 'success' status but no 'data' field or data is empty.")
                    data_preparation_status[model_name] = {"status": "ready"}
                else:
                    print(f"[DataPrep][Error] InfluxDB response status not 'success': {influx_response}")
                    data_preparation_status[model_name] = {"status": "failed", "error": influx_response}
        except Exception as e:
            print(f"[DataPrep][Exception] Data preparation failed for model {model_name}: {e}")
            data_preparation_status[model_name] = {"status": "failed", "error": str(e)}

    asyncio.create_task(prepare_data())
    return {"status": "data_preparation_started", "model": model_name}

@app.get("/data-preparation-status/{model_name}")
async def get_data_preparation_status(model_name: str):
    status = data_preparation_status.get(model_name, {"status": "unknown"})
    return status

@app.post("/start-training")
async def start_training(request: Request):
    settings = await request.json()
    model_name = settings.get("model_name")
    print(f"Starting training for model: {model_name}")
    print("Training settings:", settings)

    # Setze den initialen Trainingsstatus
    training_status[model_name] = {
        "status": "training",
        "current_epoch": 0,
        "current_loss": 0.0,
        "validation_loss": None,
        "epoch_progress": 0.0
    }

    async def run_training():
        try:
            # Validation-Set auslesen
            validation_set = settings.get('validation_set', {})
            settings['use_validation_set'] = validation_set.get('use_validation_set', 'Yes')
            settings['validation_split'] = validation_set.get('validation_split', 0.15)
            # Lade die Trainingsdaten
            data_path = Path("training_data/lstm") / f"{model_name}_data.csv"
            print(f"Looking for training data at: {data_path}")
            if not data_path.exists():
                training_status[model_name]["status"] = "failed"
                training_status[model_name]["error"] = "Training data not found"
                return
            data = pd.read_csv(data_path)
            print("Data head:")
            print(data.head())
            print("Data columns:", data.columns.tolist())
            print("Features:", settings["features"])
            print("Outputs:", settings["output"])
            trainer = LSTMTrainer(settings)
            # Start training - preprocessor wird in train() initialisiert
            training_history = trainer.train(data)
            # Speichere die Trainingshistorie
            # Get the timestamp from the trainer's last saved model directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            # Check if running in container (results mounted as /app/results) or local development
            if Path("/app/results").exists():
                history_path = Path("/app/results/trained_models/lstm") / f"{timestamp}_lstm" / f"training_history_{timestamp}_lstm.json"
            else:
                history_path = Path("results/trained_models/lstm") / f"{timestamp}_lstm" / f"training_history_{timestamp}_lstm.json"
            history_path.parent.mkdir(parents=True, exist_ok=True)
            with open(history_path, "w") as f:
                json.dump(training_history, f)
            # Aktualisiere den Trainingsstatus
            training_status[model_name]["status"] = "completed"
            training_status[model_name]["current_epoch"] = len(training_history.get("train_loss", []))
            training_status[model_name]["current_loss"] = training_history.get("train_loss", [None])[-1]
            if "val_loss" in training_history and training_history["val_loss"]:
                training_status[model_name]["validation_loss"] = training_history["val_loss"][-1]
        except Exception as e:
            print(f"Training failed: {str(e)}")
            training_status[model_name]["status"] = "failed"
            training_status[model_name]["error"] = str(e)
        # Hinweis: Die UI sollte nach dem Starten des Trainings sofort die SSE-Verbindung aufbauen, um den Fortschritt zu sehen.

    # Starte das Training im Hintergrund
    asyncio.create_task(run_training())
    return {"status": "started", "model": model_name}

@app.get("/training-status/{model_name}")
async def get_training_status(model_name: str):
    print(f"SSE connection requested for model: {model_name}", file=sys.stderr)
    
    async def event_generator():
        while True:
            if model_name in training_status:
                status = training_status[model_name]
                print(f"Current status for {model_name}: {status}", file=sys.stderr)
                
                # Format the status as a proper SSE event
                event = {
                    "event": "training_status",
                    "data": json.dumps(status),
                    "id": str(status.get("current_epoch", 0)),
                    "retry": 500
                }
                print(f"Sending event: {event}", file=sys.stderr)
                yield event
                
                if status["status"] in ["completed", "failed"]:
                    print(f"Training {status['status']} for {model_name}", file=sys.stderr)
                    break
            await asyncio.sleep(0.5)  # Update every 500ms
    
    return EventSourceResponse(event_generator())

# Callback function for training progress
def update_training_status(
    model_name: str,
    epoch: int,
    loss: float,
    epoch_progress: float = 0.0,
    val_loss: Optional[float] = None,
    progress_info: Optional[str] = None
):
    # Status-Update nur alle 5 Sekunden
    now = time.time()
    last_update = training_status[model_name].get('last_update', 0) if model_name in training_status else 0
    if now - last_update < 5 and epoch_progress < 100:
        return
    if model_name not in training_status:
        training_status[model_name] = {
            "status": "training",
            "current_epoch": 0,
            "current_loss": 0.0,
            "validation_loss": None,
            "epoch_progress": 0.0
        }
    training_status[model_name].update({
        "current_epoch": epoch,
        "current_loss": loss,
        "epoch_progress": epoch_progress,
        "validation_loss": val_loss,
        "progress_info": progress_info,
        "status": "training",
        "last_update": now
    })