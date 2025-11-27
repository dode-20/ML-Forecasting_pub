# InfluxDB API Service

FastAPI-based microservice for data access, validation, and preprocessing. This service provides a centralized interface for retrieving and processing time series data from InfluxDB for ML model training and forecasting.

## Overview

The InfluxDB API Service is designed as a **microservice** that handles all data-related operations for the ML-Forecasting system. It provides robust data retrieval, comprehensive validation, and preprocessing capabilities.

## Features

### Data Access
- **InfluxDB Integration** - Direct connection to InfluxDB time series database
- **Chunked Data Retrieval** - Efficient handling of large time ranges (30-day chunks)
- **Extended Timeouts** - 60-minute timeout for complex queries
- **Authentication Support** - Token-based authentication with InfluxDB
- **Connection Pooling** - Optimized database connections

### Data Validation
- **Multi-layer Validation** - Comprehensive data quality assurance
- **Range Validation** - Physical limits validation for PV systems
- **Outlier Detection** - Statistical outlier identification using Z-score method
- **Module Failure Detection** - Relative analysis between modules
- **Duplicate Handling** - Automatic aggregation of duplicate entries
- **Missing Data Handling** - Interpolation and gap filling strategies

### Data Preprocessing
- **Feature Engineering** - Time-based features and transformations
- **Data Normalization** - Scaling and normalization for ML models
- **Sequence Creation** - Time series sequence generation for LSTM models
- **Data Splitting** - Training, validation, and test set creation
- **Format Conversion** - CSV export and data format standardization

## API Endpoints

### Data Retrieval
```http
POST /get-training-data
Content-Type: application/json

{
    "date_selection": {
        "start": "2024-01-01",
        "end": "2024-12-31"
    },
    "module_type": "silicon",
    "selected_modules": ["module1", "module2"],
    "features": ["P", "U", "I", "Temp", "Irr"]
}
```

### Data Validation
```http
POST /validate-data
Content-Type: application/json

{
    "data_path": "/path/to/data.csv",
    "validation_config": {
        "outlier_threshold": 3.0,
        "range_validation": true,
        "module_failure_detection": true
    }
}
```

### Health Check
```http
GET /health
```

## Technical Architecture

### Framework
- **FastAPI** - Modern, fast web framework for building APIs
- **InfluxDB Client** - Official InfluxDB Python client
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Pydantic** - Data validation using Python type annotations

### Data Processing Pipeline
```python
class DataProcessor:
    def __init__(self, influx_config):
        self.client = InfluxDBClient(influx_config)
        self.validator = DataValidator()
        self.preprocessor = DataPreprocessor()
    
    async def get_training_data(self, request):
        # 1. Query InfluxDB
        raw_data = await self.client.query_data(request)
        
        # 2. Validate data
        validated_data = await self.validator.validate(raw_data)
        
        # 3. Preprocess data
        processed_data = await self.preprocessor.preprocess(validated_data)
        
        return processed_data
```

### Environment Detection
```python
# Automatic environment detection
def detect_environment():
    if os.getenv('DOCKER_ENV') or os.path.exists('/.dockerenv'):
        return 'docker'
    elif os.getenv('OFFLINE_MODE'):
        return 'offline'
    else:
        return 'offline'  # Default to offline for safety
```

## Configuration

### Environment Variables
```bash
# InfluxDB Configuration
INFLUX_URL=https://your-influxdb-hostname:8086
INFLUX_TOKEN=your-influxdb-token
INFLUX_ORG=your-organization
INFLUX_BUCKET=your-bucket

# Service Configuration
API_HOST=0.0.0.0
API_PORT=8009
LOG_LEVEL=INFO

# Data Processing
CHUNK_SIZE_DAYS=30
MAX_TIMEOUT_MINUTES=60
OUTLIER_THRESHOLD=3.0
```

### Docker Configuration
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root

COPY . .
CMD ["poetry", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8009"]
```

## Installation and Setup

### Local Development
```bash
cd influxData_api
poetry install
poetry run uvicorn main:app --host 0.0.0.0 --port 8009
```

### Docker Deployment
```bash
# With Docker Compose (recommended)
docker-compose up api_influxdata

# Or directly
docker build -t influxdata-api .
docker run -p 8009:8009 influxdata-api
```

## Data Validation System

### Validation Layers

1. **Range Validation**
   - Temperature: -40°C to 100°C (module), -40°C to 60°C (ambient)
   - Humidity: 0% to 100%
   - Irradiance: 0 to 1200 W/m²
   - Electrical parameters: Voltage (0-1000V), Current (0-20A), Power (0-500W)

2. **Outlier Detection**
   - Z-score method with configurable threshold (default: 3.0 standard deviations)
   - Statistical outlier identification based on distribution analysis
   - Average mode for silicon and perovskite modules

3. **Module Failure Detection**
   - Relative analysis comparing modules to each other
   - Detects modules deviating significantly from the group (3+ standard deviations)
   - Sustained deviation analysis over multiple consecutive hours
   - Correlation-based failure detection

4. **Duplicate Handling**
   - Automatic detection and aggregation of duplicate entries
   - Aggregates numeric columns using mean values
   - Preserves first value for non-numeric columns

### Validation Output
```
validation_run_YYYYMMDD_HHMMSS/
├── 00_duplicates_debug_*.csv      # Duplicate entries analysis
├── 00_duplicates_summary_*.csv    # Duplicate patterns summary
├── 01_range_violations_*.csv      # Data outside physical limits
├── 02_outliers_*.csv              # Statistical outliers
├── 03_module_failures_*.csv       # Module failure instances
├── 04_validation_summary_*.csv    # Comprehensive statistics
└── 05_FINAL_CLEANED_DATA_*.csv    # Final validated dataset
```

## Integration with Other Services

### Dashboard Integration
- Provides data for training interface
- Real-time data validation status
- Configuration management for data processing

### Model Services Integration
- Supplies validated training data
- Provides preprocessing configurations
- Handles data format requirements for different models

### Offline Testing Integration
- Unified architecture supports both Docker and offline modes
- Automatic fallback to sample data in offline mode
- Consistent API interface regardless of environment


## Status

**Current Status**: Partially implemented
- Basic InfluxDB integration
- Core data validation system
- API endpoints defined
- Docker containerization
- Extended validation features (in development)
- Advanced preprocessing (in development)

The service provides a solid foundation for data access and validation but can be further developed to integrate all features of the comprehensive offline testing framework.
