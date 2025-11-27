# Model Manager Service

FastAPI-based microservice for model lifecycle management, service discovery, and load balancing across multiple ML model services. This service acts as a central orchestrator for managing and routing requests to different model services.

## Overview

The Model Manager Service is designed as a **microservice** that provides centralized management of ML models and their services. It handles service discovery, load balancing, model lifecycle management, and provides a unified interface for model operations.

## Features

### Service Discovery
- **Automatic Service Detection** - Discovers available model services
- **Health Monitoring** - Continuous health checks of model services
- **Service Registration** - Dynamic registration of new model services
- **Service Metadata** - Tracks service capabilities and configurations

### Load Balancing
- **Request Distribution** - Distributes requests across available model services
- **Round-robin Routing** - Simple round-robin load balancing
- **Service Selection** - Intelligent service selection based on availability
- **Failover Support** - Automatic failover to healthy services

### Model Lifecycle Management
- **Model Deployment** - Manages model deployment across services
- **Version Management** - Tracks model versions and updates
- **Model Switching** - Dynamic model switching without service restart
- **Model Retirement** - Graceful model retirement and cleanup

### Unified API
- **Single Entry Point** - Unified API for all model operations
- **Request Routing** - Intelligent routing to appropriate services
- **Response Aggregation** - Aggregates responses from multiple services
- **Error Handling** - Centralized error handling and reporting

## API Endpoints

### Service Management
```http
# Get all available services
GET /services

# Get service health status
GET /services/{service_id}/health

# Register new service
POST /services/register
Content-Type: application/json
{
    "service_id": "lstm_model_v2",
    "service_url": "http://lstm_model:8000",
    "model_type": "LSTM",
    "capabilities": ["training", "prediction"]
}
```

### Model Operations
```http
# Unified prediction endpoint
POST /predict
Content-Type: application/json
{
    "model_type": "LSTM",
    "data": [0.0, 0.01, 0.02, ...],
    "model_version": "latest"
}

# Unified training endpoint
POST /train
Content-Type: application/json
{
    "model_type": "LSTM",
    "config": {...}
}
```

### Health Check
```http
GET /health
```

## Technical Architecture

### Framework
- **FastAPI** - Modern, fast web framework for building APIs
- **HTTPX** - Async HTTP client for service communication
- **Pydantic** - Data validation and serialization
- **Asyncio** - Asynchronous programming support


## Configuration

### Environment Variables
```bash
# Service Configuration
API_HOST=0.0.0.0
API_PORT=8008
LOG_LEVEL=INFO

# Service Discovery
SERVICE_DISCOVERY_INTERVAL=30
HEALTH_CHECK_INTERVAL=10
SERVICE_TIMEOUT=30

# Load Balancing
LOAD_BALANCING_STRATEGY=round_robin
MAX_RETRIES=3
RETRY_DELAY=1
```

### Service Configuration
```json
{
    "services": {
        "lstm_model": {
            "service_url": "http://lstm_model:8000",
            "model_type": "LSTM",
            "capabilities": ["training", "prediction"],
            "max_concurrent_requests": 5,
            "timeout": 30
        },
        "lstm_multistep_model": {
            "service_url": "http://lstm_multistep:8001",
            "model_type": "LSTM_MULTISTEP",
            "capabilities": ["training", "multistep_prediction"],
            "max_concurrent_requests": 3,
            "timeout": 60
        }
    }
}
```

### Docker Configuration
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root

COPY . .
CMD ["poetry", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8008"]
```

## Installation and Setup

### Local Development
```bash
cd model_manager
poetry install
poetry run uvicorn main:app --host 0.0.0.0 --port 8008
```

### Docker Deployment
```bash
# With Docker Compose (recommended)
docker-compose up model_manager

# Or directly
docker build -t model-manager .
docker run -p 8008:8008 model-manager
```


## Integration with Other Services

### Dashboard Integration
- Service status monitoring
- Model management interface
- Load balancing statistics
- Health monitoring dashboard

### Model Services Integration
- Automatic service registration
- Health check coordination
- Request routing and forwarding
- Error handling and failover

### InfluxDB API Integration
- Data service coordination
- Training data routing
- Validation service integration



## Status

**Current Status**: Partially implemented
- Basic service discovery mechanism
- Simple load balancing implementation
- API endpoints defined
- Docker containerization
- Advanced monitoring (in development)
- Circuit breaker patterns (in development)
- Comprehensive metrics collection (in development)

The service provides a foundation for model management and orchestration but can be further developed to integrate all features of the comprehensive offline testing framework and provide advanced service mesh capabilities.
