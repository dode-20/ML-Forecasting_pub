from fastapi import FastAPI, HTTPException, Request
import requests
from pydantic import BaseModel
import docker
from fastapi.responses import StreamingResponse, Response
import httpx
from httpx import StreamClosed

app = FastAPI()

# Mapping of available models and their API endpoints
MODEL_ENDPOINTS = {
    "lstm": "http://lstm_model:8001",
    # future additional models
}

MODEL_SERVICE_MAP = {
    "LSTM": "http://lstm_model:8000",
    "CNN": "http://cnn_model:8001"
    # Add additional models here
}

# Global map for external training URLs
EXTERNAL_URL_MAP = {}

def get_model_service_url(model_type: str) -> str:
    if model_type in MODEL_SERVICE_MAP:
        return MODEL_SERVICE_MAP[model_type]
    raise HTTPException(status_code=400, detail=f"Unknown model type: {model_type}")

@app.post("/predict/{model_name}")
async def predict(model_name: str, payload: dict):
    if model_name not in MODEL_ENDPOINTS:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not registered.")

    try:
        response = requests.post(MODEL_ENDPOINTS[model_name], json=payload)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error forwarding request: {str(e)}")

@app.post("/train/{model_name}")
async def train(model_name: str, payload: dict):
    if model_name not in MODEL_ENDPOINTS:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not registered.")

    train_url = MODEL_ENDPOINTS[model_name].replace("/predict", "/train")
    try:
        response = requests.post(train_url, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error forwarding training request: {str(e)}")

docker_client = docker.from_env()
LSTM_CONTAINER_NAME = "ml-forecasting-lstm_model-1"

class ControlRequest(BaseModel):
    action: str  # "start" or "stop"

@app.get("/status/lstm")
def get_lstm_status():
    try:
        container = docker_client.containers.get(LSTM_CONTAINER_NAME)
        return {"running": container.status == "running"}
    except docker.errors.NotFound:
        raise HTTPException(status_code=404, detail="LSTM container not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/control/lstm")
def control_lstm_container(req: ControlRequest):
    try:
        container = docker_client.containers.get(LSTM_CONTAINER_NAME)
        if req.action == "start" and container.status != "running":
            container.start()
        elif req.action == "stop" and container.status == "running":
            container.stop()
        else:
            return {"message": f"Container already {container.status}"}
        return {"message": f"Container {req.action}ed"}
    except docker.errors.NotFound:
        raise HTTPException(status_code=404, detail="LSTM container not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "running", "service": "model_manager"}

@app.post("/train_model")
async def train_model(request: Request):
    settings = await request.json()
    model_type = settings.get("model_type")
    model_name = settings.get("model_name")
    # Dynamic service URL for external training
    ext = settings.get("external_training", {})
    if ext.get("enabled") and ext.get("url"):
        base_url = ext["url"].rstrip("/")
        EXTERNAL_URL_MAP[model_name] = base_url
    else:
        base_url = get_model_service_url(model_type)
    async with httpx.AsyncClient() as client:
        # Step 1: get-training-data
        data_resp = await client.post(base_url + "/get-training-data", json=settings)
        if data_resp.status_code != 200:
            return Response(content=data_resp.content, status_code=data_resp.status_code)
        # Step 2: start-training
        train_resp = await client.post(base_url + "/start-training", json=settings)
        return Response(content=train_resp.content, status_code=train_resp.status_code)

@app.get("/training-status/{model_name}")
async def proxy_training_status(model_name: str, model_type: str):
    # Check if an external URL is stored for this model:
    base_url = EXTERNAL_URL_MAP.get(model_name, get_model_service_url(model_type))
    target_url = base_url + f"/training-status/{model_name}"
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("GET", target_url, headers={"Accept": "text/event-stream"}) as response:
            async def event_generator():
                try:
                    async for line in response.aiter_lines():
                        if line.strip() == "":
                            continue
                        yield f"{line}\n"
                except StreamClosed:
                    # Stream wurde vom Backend geschlossen, das ist ok
                    return
            return StreamingResponse(event_generator(), media_type="text/event-stream")