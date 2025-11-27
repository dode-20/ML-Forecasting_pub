# Dashboard Service

Streamlit-based user interface for the ML-Forecasting system. The dashboard provides a central interface for all main system functions.

## Overview

The dashboard is designed as a **microservice** and offers multiple pages for different aspects of the ML-Forecasting system.

## Pages

### 1. General Dashboard (`1_general_dashboard.py`)
- **System Status Monitoring** - Monitoring of all container services
- **Container Health Checks** - Status of Docker services
- **Quick Access** - Quick access to all features
- **Service Discovery** - Automatic detection of available services

### 2. Training Interface (`pages/2_training.py`)
- **Model Configuration** - Configuration of model parameters
- **Real-time Training Progress** - Live training with Server-Sent Events (SSE)
- **Data Preprocessing** - Data preparation and validation
- **Model Performance Metrics** - Training metrics and visualizations

### 3. Forecast Overview (`pages/3_forecast_overview.py`)
- **Historical Forecast Analysis** - Analysis of past forecasts
- **Model Comparison** - Comparison of different models
- **Performance Visualization** - Detailed performance displays
- **Forecast Accuracy Metrics** - Accuracy metrics

### 4. Documentation (`pages/4_documentation.py`)
- **API Documentation** - Complete API documentation
- **Usage Examples** - Application examples and tutorials
- **Configuration Guides** - Configuration guides

## Features

### Service Integration
- **FastAPI Services** - Integration with all backend services
- **Real-time Communication** - Server-Sent Events for live updates
- **Error Handling** - Robust error handling for service failures
- **Fallback Mechanisms** - Automatic fallbacks for service issues

### User Interface
- **Responsive Design** - Adaptation to different screen sizes
- **Interactive Visualizations** - Plotly-based interactive diagrams
- **Progress Tracking** - Real-time progress displays
- **Configuration Management** - JSON-based configuration management

## Technical Details

### Framework
- **Streamlit** - Web-based Python application
- **Plotly** - Interactive visualizations
- **Pandas** - Data processing and analysis
- **Requests** - HTTP client for API communication

### Service Communication
```python
# Service Discovery
SERVICES = {
    'influxdata_api': 'http://api_influxdata:8009',
    'model_manager': 'http://model_manager:8008',
    'lstm_model': 'http://lstm_model:8000'
}

# Health Check
async def check_service_health(service_url):
    try:
        response = requests.get(f"{service_url}/health", timeout=5)
        return response.status_code == 200
    except:
        return False
```

### Real-time Updates
```python
# Server-Sent Events for Training Progress
def stream_training_progress(model_name):
    # SSE Implementation for live updates
    pass
```

## Configuration

### Environment Variables
```bash
# Dashboard Configuration
STREAMLIT_SERVER_PORT=8502
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_WATCHER_TYPE=none

# Service URLs
INFLUXDATA_API_URL=http://api_influxdata:8009
MODEL_MANAGER_URL=http://model_manager:8008
LSTM_MODEL_URL=http://lstm_model:8000
```

### Docker Configuration
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root

COPY . .
CMD ["poetry", "run", "streamlit", "run", "1_general_dashboard.py", \
     "--server.port=8502", "--server.address=0.0.0.0"]
```

## Installation and Setup

### Local Development
```bash
cd dashboard
poetry install
poetry run streamlit run 1_general_dashboard.py
```

### Docker Deployment
```bash
# With Docker Compose (recommended)
docker-compose up dashboard_main

# Or directly
docker build -t dashboard .
docker run -p 8502:8502 dashboard
```

## API Integration

The dashboard communicates with the following services:

### InfluxDB API Service
- **Data Retrieval** - Retrieving training data
- **Data Validation** - Data quality checks
- **Configuration Management** - Model configurations

### Model Manager Service
- **Model Lifecycle** - Model creation, training, deployment
- **Service Discovery** - Automatic detection of available models
- **Load Balancing** - Distribution of requests across different models

### LSTM Model Services
- **Training** - Model training with live progress
- **Prediction** - Prediction generation
- **Model Management** - Saving and loading models


## Extensibility

The dashboard is modularly structured and can be easily extended:

### Adding New Pages
1. Create new file in `pages/`
2. Add navigation in `1_general_dashboard.py`
3. Implement service integration

### New Visualizations
- Plotly-based diagrams for new metrics
- Custom components for special displays
- Export functions for reports

### API Extensions
- New endpoints for additional services
- Caching mechanisms for better performance
- Authentication and authorization

## Status

**Current Status**: Partially implemented
- Basic dashboard structure
- Service integration
- Real-time updates
- Extended visualizations (in development)
- Complete API integration (in development)

The dashboard provides a solid foundation for the microservices architecture but can be further developed to integrate all features of the offline testing framework.