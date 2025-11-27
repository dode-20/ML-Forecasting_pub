# Multi-Step LSTM Model Service

Extended LSTM model for **Multi-Step-Ahead Predictions** with optional **Weather-Informed** functionality. This service provides containerized multi-step LSTM models for photovoltaic energy forecasting.

## LSTM Model Implementation

### Architecture Overview
The multi-step LSTM model predicts multiple future timesteps of PV energy output. It can operate in two modes:
- **Historical-only**: Uses only historical data for predictions
- **Weather-Informed**: Incorporates weather forecast data for improved accuracy

### Model Architecture
```python
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, forecast_steps):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_steps = forecast_steps
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        # Take the last output
        last_output = lstm_out[:, -1, :]
        # Apply dropout and linear layer
        output = self.fc(self.dropout(last_output))
        return output
```

### Implementation Process

#### 1. Data Preprocessing (`service/src/preprocess.py`)
- **Multi-Step Sequence Creation**: Creates sequences for multiple future timesteps
- **Weather Data Integration**: Incorporates weather forecast data when available
- **Normalization**: Normalizes features for stable training

#### 2. Model Training (`service/src/train.py`)
- **Multi-Step Trainer**: Handles training for multiple output timesteps
- **Weather-Informed Mode**: Automatically detects and uses weather forecast data
- **Sequence Handling**: Manages both historical and forecast data sequences

#### 3. Model Inference (`service/src/model.py`)
- **Multi-Step Prediction**: Generates predictions for multiple future timesteps
- **Weather Integration**: Uses weather forecast data for improved predictions
- **Output Formatting**: Returns structured multi-step predictions

#### 4. API Interface (`api/routes.py`)
- **Multi-Step Endpoint**: `/predict-multistep` - Returns multi-step predictions
- **Weather Integration**: Automatic weather data integration when available

### Key Differences from One-Step Model

#### 1. Model Architecture
- **One-Step**: `output_size = 1` (single next timestep)
- **Multi-Step**: `output_size = forecast_steps` (multiple future timesteps)

#### 2. Sequence Creation
- **One-Step**: `Input[t:t+seq_len] -> Target[t+seq_len-1]`
- **Multi-Step (Historical-only)**: `Input[t:t+seq_len] -> Target[t+seq_len:t+seq_len+forecast_steps]`
- **Multi-Step (Weather-Informed)**: `Input[t:t+seq_len+forecast_steps] -> Target[t+seq_len:t+seq_len+forecast_steps]`
  - **Historical**: `[t:t+seq_len]` - historical features
  - **Forecast**: `[t+seq_len:t+seq_len+forecast_steps]` - weather forecast features (optional)
  - **Output**: `[t+seq_len:t+seq_len+forecast_steps]` - P_normalized predictions

#### 3. API Endpoints
- **One-Step**: `/predict` - returns `List[float]`
- **Multi-Step**: `/predict-multistep` - returns `List[List[float]]`

## Configuration

### Multi-Step Configuration
```json
{
    "forecast_mode": {
        "mode": "multi-step",
        "forecast_steps": 6
    },
    "weather_data": {
        "use_weatherData": true,
        "weather_features": ["TT_10", "RF_10", "RWS_10", "RWS_IND_10", "V_N"]
    },
    "data_resolution": "1h",
    "model_type": "LSTM_MULTISTEP"
}
```

### Weather-Informed Mode
Automatically activated when:
- `weather_data.use_weatherData = true`
- `weather_data.weather_features` configured
- Weather forecast data available during training

**Forecast Horizon**: `forecast_steps × data_resolution`

### Usage Examples

#### 6-Hour Forecast (1h resolution) - Weather-Informed:
```json
{
    "sequence_length": 8,
    "forecast_mode": {
        "mode": "multi-step",
        "forecast_steps": 6
    },
    "weather_data": {
        "use_weatherData": true,
        "weather_features": ["TT_10", "RF_10", "RWS_10", "RWS_IND_10", "V_N"]
    },
    "data_resolution": "1h"
}
// Forecast Horizon: 6 × 1h = 6 hours
// Input: 8 historical + 6 weather forecast = 14 timesteps
// Output: 6 P_normalized predictions
```

#### 24-Hour Forecast (1h resolution):
```json
{
    "sequence_length": 24,
    "forecast_mode": {
        "mode": "multi-step",
        "forecast_steps": 24
    },
    "data_resolution": "1h"
}
// Forecast Horizon: 24 × 1h = 24 hours
```

### Advantages of Weather-Informed Multi-Step Models
1. **Efficiency**: Single model call for multiple prediction steps
2. **Consistency**: All predictions based on same input sequence
3. **Speed**: Faster than iterative one-step predictions
4. **Realistic Predictions**: Uses actual weather forecasts
5. **Better Performance**: Exogenous variables (weather) are highly predictive
6. **Practical**: Weather forecasts are available and reliable

### Recommended Usage
- **Short-term (1-6h)**: Weather-Informed Multi-Step Model
- **Medium-term (6-24h)**: Weather-Informed Multi-Step Model
- **Long-term (1-7 days)**: Weather-Informed Multi-Step Model with longer sequences

## File Structure

```
lstm_model_multistep/
├── service/src/
│   ├── model.py              # Extended LSTM architecture
│   ├── preprocess.py         # Multi-step sequence creation
│   └── train.py              # Training logic
├── api/
│   ├── models.py             # Multi-step API models
│   └── routes.py             # Multi-step endpoints
├── example_multistep_config.json
├── test_multistep_training.py
└── README_MULTISTEP.md
```

---

## FastAPI Container Service

### API Endpoints
- **Multi-Step Prediction**: `POST /predict-multistep` - Get multi-step predictions
- **Training**: `POST /train` - Train multi-step LSTM model
- **Training Status**: `GET /training-status/{model_name}` - Real-time training progress (SSE)
- **Health Check**: `GET /health` - Service health status

### Docker Deployment
```bash
# Local Development
cd lstm_model_multistep
poetry install
poetry run uvicorn service.main:app --host 0.0.0.0 --port 8000

# Docker Compose
docker-compose up lstm_model_multistep

# Direct Docker
docker build -t lstm-model-multistep .
docker run -p 8000:8000 lstm-model-multistep
```

### Environment Variables
```bash
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
MODEL_STORAGE_PATH=/app/models
TRAINING_TIMEOUT=3600
MAX_CONCURRENT_TRAINING=1
```

### Status
**Current Status**: Partially implemented
- Multi-step LSTM model architecture [OK]
- Weather-informed functionality [OK]
- API endpoints defined [OK]
- Docker containerization [OK]
- Real-time progress monitoring (in development)
- Advanced model management (in development)