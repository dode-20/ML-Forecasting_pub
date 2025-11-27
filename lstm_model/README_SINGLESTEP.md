# Single-Step LSTM Model Service

FastAPI-based microservice for single-step LSTM model training and prediction. This service provides a containerized LSTM model for photovoltaic energy forecasting with real-time training progress monitoring.

## LSTM Model Implementation

### Architecture Overview
The single-step LSTM model is designed to predict the next timestep of PV energy output. It uses a sequence of historical data points to forecast the immediate future value.

### Model Architecture
```python
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
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
- **Sequence Creation**: Creates time series sequences from raw data
- **Normalization**: Normalizes features for stable training
- **Train/Validation Split**: Splits data for training and validation

#### 2. Model Training (`service/src/train.py`)
- **LSTMTrainer Class**: Handles the complete training pipeline
- **Training Loop**: Implements epoch-based training with validation
- **Early Stopping**: Prevents overfitting with validation monitoring
- **Model Saving**: Saves trained models with metadata

#### 3. Model Inference (`service/src/model.py`)
- **LSTMModel Class**: Defines the neural network architecture
- **Forward Pass**: Implements the LSTM forward propagation
- **Prediction**: Generates single-step predictions

#### 4. API Interface (`api/routes.py`)
- **Training Endpoint**: `/train` - Initiates model training
- **Prediction Endpoint**: `/predict` - Returns single-step predictions
- **Status Endpoint**: `/training-status/{model_name}` - Real-time training progress

### Key Features
- **Single-step Predictions**: Next timestep PV energy forecasting
- **Real-time Training Progress**: Live monitoring with Server-Sent Events (SSE)
- **Automatic Model Saving**: Timestamped model artifacts with configuration preservation
- **Early Stopping**: Validation-based early stopping to prevent overfitting
- **Model Versioning**: Automatic model versioning with metadata

## Configuration

### Model Configuration
```json
{
    "model_name": "lstm_silicon_v1",
    "model_type": "LSTM",
    "input_size": 5,
    "hidden_size": 128,
    "num_layers": 3,
    "output_size": 1,
    "sequence_length": 24,
    "features": ["P", "U", "I", "Temp", "Irr"],
    "output": ["P_normalized"],
    "training": {
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 0.001,
        "shuffle": true,
        "loss_function": "MSE"
    }
}
```

### Training Process
1. **Data Loading** - Load training data from InfluxDB API
2. **Sequence Creation** - Create time series sequences for LSTM
3. **Normalization** - Normalize features for stable training
4. **Train/Validation Split** - Split data for training and validation
5. **Training Loop** - Epoch-based training with early stopping
6. **Model Saving** - Save trained models with metadata

### Model Storage Structure
```
models/
├── YYYYMMDD_HHMM_lstm_silicon_v1/
│   ├── YYYYMMDD_HHMM_lstm_silicon_v1.pth      # Model weights
│   ├── model_config_YYYYMMDD_HHMM_lstm_silicon_v1.json  # Configuration
│   ├── training_history_YYYYMMDD_HHMM_lstm_silicon_v1.json  # Training metrics
│   └── training_summary_YYYYMMDD_HHMM_lstm_silicon_v1.txt  # Summary
└── ...
```


---

## FastAPI Container Service

### API Endpoints
- **Training**: `POST /train` - Train LSTM model with configuration
- **Prediction**: `POST /predict` - Get single-step predictions
- **Training Status**: `GET /training-status/{model_name}` - Real-time training progress (SSE)
- **Health Check**: `GET /health` - Service health status

### Docker Deployment
```bash
# Local Development
cd lstm_model
poetry install
poetry run uvicorn service.main:app --host 0.0.0.0 --port 8000

# Docker Compose
docker-compose up lstm_model

# Direct Docker
docker build -t lstm-model .
docker run -p 8000:8000 lstm-model
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
- Basic LSTM model architecture [OK]
- Core training and prediction functionality [OK]
- API endpoints defined [OK]
- Docker containerization [OK]
- Real-time progress monitoring (in development)
- Advanced model management (in development)
