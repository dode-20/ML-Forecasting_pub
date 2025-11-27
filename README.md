# PV Forecasting Tool with ML Models

A comprehensive system for training, evaluating, and deploying machine learning models to forecast photovoltaic (PV) energy yields. The project combines an originally planned microservices architecture with a fully functional offline framework.

## Project Overview

The project was conceived as a **microservices-based architecture** but was primarily developed as a **comprehensive offline testing framework**. The original Docker-based microservices structure is partially implemented, but the main focus lies on the extensive offline testing system.

### Current Project Structure

```
ML-forecasting/
├── dashboard/                    # Streamlit Dashboard (Microservice)
├── influxData_api/              # InfluxDB API Service (Microservice)
├── lstm_model/                  # Single-Step LSTM Model (Microservice)
├── lstm_model_multistep/        # Multi-Step LSTM Model (Microservice)
├── model_manager/               # Model Management Service (Microservice)
├── dwd_weatherData/             # German Weather Service Integration
├── offline_tests/               # Main development area
│   ├── forecast_evaluator/      # Comprehensive model evaluation
│   ├── experimental_scenarios/  # Experimental configurations
│   └── [various test scripts]
├── results/                     # Centralized results
│   ├── model_configs/          # Model configuration files
│   ├── trained_models/         # Trained model artifacts
│   ├── training_data/          # Training data and validation results
│   ├── forecast_data/          # Forecast data and results
│   ├── parameter_analysis/     # Hyperparameter analysis
│   └── irr_comparison/         # DWD vs InfluxDB irradiance comparisons
└── training_data/               # Legacy training data (deprecated)
```

## Main Features

### Fully Implemented (Offline Tests)

- **Comprehensive Offline Testing Framework** - Complete ML pipeline without Docker
- **Unified Pipeline Architecture** - Single entry point with automatic data detection
- **LSTM Model Training** - Single-Step and Multi-Step predictions with dynamic configuration
- **Enhanced Data Validation** - Automatic scaling factor integration and intelligent gap filling
- **DWD Weather Data Integration** - German Weather Service integration with comparison analysis
- **Forecast Evaluation** - Detailed model performance analysis and visualization
- **Parameter Analysis** - Systematic hyperparameter optimization
- **Perovskite vs Silicon Analysis** - Technology comparison and data availability assessment
- **Automatic Environment Detection** - Seamless Docker/Offline mode switching

### Partially Implemented (Microservices)

- **Docker-based Microservices** - Basic structure present, but not fully developed
- **Streamlit Dashboard** - Basic interface implemented
- **FastAPI Services** - API endpoints defined, but limited functionality
- **Model Manager** - Basic model management present

## Quick Start

### Recommended Approach: Offline Tests (Main Development)

The project was primarily developed as an offline testing framework and is most complete here:

```bash
# 1. Clone repository
git clone <repository-url>
cd ML-forecasting

# 2. Navigate to offline tests
cd offline_tests

# 3. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# PyTorch according to operating system:
# macOS:
pip install -r requirements-torch_mac.txt
# Windows:
pip install -r requirements-torch_windows.txt

# 5. Run complete pipeline
python lstm_pipeline_entrypoint.py
```

### Alternative: Docker Microservices (Limited Functionality)

For the originally planned microservices architecture:

```bash
# Start services
docker-compose up --build

# Access dashboard
# http://localhost:8502
```

**Note**: The Docker environment is not fully developed and has limited functionality compared to the offline testing framework.

## Component Documentation

For detailed information about specific components:

### Main Development (Offline Tests)
- **[Offline Tests](offline_tests/README.md)** - Comprehensive testing framework without Docker
- **[DWD Irr Comparison](offline_tests/dwd_weatherData/README_dwd_irr_comparison.md)** - Weather data source comparison and scaling factors
- **[Perovskite Analysis](offline_tests/pvk_data_availability/README_perovskite_analysis.md)** - Silicon vs Perovskite data availability analysis

### Microservices (Partially implemented)
- **[Multi-Step LSTM](lstm_model_multistep/README_MULTISTEP.md)** - Multi-step predictions with weather integration
- **[Dashboard](dashboard/README.md)** - Streamlit interface and visualization tools
- **[InfluxDB API](influxData_api/README.md)** - Data access and validation services
- **[Single-Step LSTM](lstm_model/README_SINGLESTEP.md)** - Single-step forecasting model
- **[Model Manager](model_manager/README.md)** - Model lifecycle management


## Results Directory Structure

The project uses a centralized `results/` structure:

```
results/
├── model_configs/               # Model configuration files
├── trained_models/              # Trained model artifacts
│   ├── lstm/                   # LSTM model outputs
│   └── offline_models/         # Offline trained models
├── training_data/              # Training data and validation results
│   ├── rawData/               # Raw data from InfluxDB
│   ├── cleanData/             # Cleaned and validated data
│   └── validation_results/    # Data validation outputs
├── forecast_data/              # Forecast data and results
├── forecast_evaluation/        # Model evaluation results
├── parameter_analysis/         # Hyperparameter analysis
├── perovskite_analysis/        # Perovskite vs Silicon analyses
├── weather_data/               # Weather data and analyses
└── irr_comparison/             # DWD vs InfluxDB irradiance comparisons
```

## Development History and Architecture

### Original Vision: Microservices Architecture

The project was conceived as a **modular microservices architecture**:

- **Dashboard Service** (Streamlit) - User interface
- **InfluxDB API Service** - Data access and validation
- **LSTM Model Services** - Single-Step and Multi-Step predictions
- **Model Manager Service** - Model lifecycle management
- **Docker Compose** - Orchestration of all services

### Actual Development: Offline Testing Framework

During development, the focus was on a **comprehensive offline testing framework**:

- **Scientifically-based data validation** with multiple validation layers
- **Complete ML pipeline** from data query to model evaluation
- **Unified Architecture** - Automatic detection of Docker/Offline environments
- **Experimental scenarios** for systematic model development
- **Forecast evaluation** with detailed performance metrics

### Current Status

- **Offline Testing Framework**: Fully implemented and functional
- **Microservices Architecture**: Basic structure present, but not fully developed
- **Unified Architecture**: Works in both environments
- **Scientific Analyses**: Perovskite analysis, DWD comparisons, parameter optimization

## Key Features

### Enhanced Data Validation and Cleaning

The system implements advanced multi-layer data validation:

- **Range Validation** - Against realistic physical limits for PV systems
- **Outlier Detection** - Z-score method with configurable thresholds
- **Module Failure Detection** - Relative analysis between modules
- **Duplicate Handling** - Automatic aggregation of duplicates
- **Time Features** - Integration of temporal patterns for improved training
- **Automatic Scaling Factor Integration** - Uses DWD comparison analysis for intelligent gap filling
- **Weather-Informed Validation** - Leverages weather data for improved data quality

### Model Training and Evaluation

- **LSTM Models** - Single-Step and Multi-Step predictions with dynamic configuration
- **Weather-Informed Mode** - Integration of weather forecast data with automatic scaling
- **Unified Pipeline** - Single entry point with automatic data detection and processing
- **Parameter Analysis** - Systematic hyperparameter optimization
- **Performance Metrics** - Comprehensive model evaluation with detailed visualizations
- **Config-Driven Training** - All model parameters managed through centralized configuration

### Experimental Scenarios and Analysis

- **Perovskite vs Silicon** - Technology comparison and data availability assessment
- **DWD vs InfluxDB** - Weather data source comparison and scaling factor generation
- **Module Type Analyses** - Various PV module technologies with automatic detection
- **Seasonal Analyses** - Seasonal patterns and trends with weather integration
- **Scaling Factor Analysis** - Automatic generation and integration of data validation scaling factors

## Technical Details

### Environment Detection

The system automatically detects the operating environment:

```python
# Docker Environment Detection
- DOCKER_ENV environment variable
- KUBERNETES_SERVICE_HOST environment variable
- /.dockerenv file presence
- Docker Compose environment variables

# Offline Environment Detection
- OFFLINE_MODE=true environment variable
- Local development environment
- Missing Docker indicators
```

### Unified Architecture and Pipeline

- **Unified codebase** for Docker and Offline environments
- **Single Pipeline Entry Point** - `lstm_pipeline_entrypoint.py` for complete workflow
- **Automatic Data Detection** - Scripts automatically find and use available data
- **Config-Driven Processing** - All processing steps use centralized configuration
- **Automatic fallbacks** to sample data in Offline mode
- **Consistent results structure** in both environments
- **Seamless transitions** between modes
- **Error Recovery** - Robust error handling and recovery mechanisms

## Next Steps and Extensibility

### Expand Microservices Architecture

The originally planned microservices architecture can be further developed:

- **Complete API implementation** for all services
- **Enhanced dashboard functionality** with all offline test features
- **Kubernetes deployment** for production environments
- **Monitoring and logging** for all services

### Extend Offline Testing Framework

- **Additional ML models** (XGBoost, LightGBM, CNN)
- **Extended evaluation metrics** and visualization tools
- **Automated hyperparameter optimization** with advanced search algorithms
- **Real-time model monitoring** and performance tracking
- **Enhanced weather data integration** with additional weather services
- **Advanced scaling factor analysis** for different weather conditions and seasons

## Installation and Setup

### Poetry Dependency Management

The project uses [Poetry](https://python-poetry.org/) for dependency management:

```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Run scripts
poetry run python script.py
```

### Environment Variables

```bash
# InfluxDB Configuration
INFLUX_URL=https://your-influxdb-hostname:8086
INFLUX_TOKEN=your-token
INFLUX_ORG=your-org
INFLUX_BUCKET=your-bucket

# Environment Mode
DOCKER_ENV=true          # Docker mode
OFFLINE_MODE=true        # Offline mode
```

