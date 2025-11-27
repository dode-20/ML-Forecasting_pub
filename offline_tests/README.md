# Offline Tests

This directory contains offline tests for the ML-Forecasting system that can be executed without Docker containers.

## Overview

The tests verify various components of the system:
- **InfluxDB Data Query**: Tests data queries from InfluxDB
- **Model Training**: Tests LSTM model training
- **Forecast Prediction**: Tests prediction functionality

## Unified Architecture

The system now uses a **unified architecture** that supports both Docker and offline modes:

### How It Works

1. **Environment Detection**: The system automatically detects whether it's running in Docker or offline mode
2. **Single Codebase**: Both modes use the same code, eliminating duplication
3. **Backward Compatibility**: Docker environment continues to work exactly as before
4. **Offline Fallbacks**: When InfluxDB is unavailable, realistic sample data is generated

### Environment Detection

The system detects the mode using these criteria (in order):
1. `DOCKER_ENV` or `KUBERNETES_SERVICE_HOST` environment variables
2. `OFFLINE_MODE=true` environment variable
3. Presence of `/.dockerenv` file
4. Defaults to offline mode for safety

### Mode-Specific Behavior

**Docker Mode:**
- Connects to real InfluxDB
- Uses API status updates
- Saves models to `models/` directory
- Full SSE (Server-Sent Events) support

**Offline Mode:**
- Generates realistic sample data when InfluxDB unavailable
- No API status updates (console output only)
- Saves models to `offline_models/` directory
- Simplified logging

## Offline Test Setup Workflow

To run the offline tests, please follow these steps:

### 1. Create and activate a virtual environment

Navigate to the `offline_tests` directory and create a new virtual environment:

```sh
cd offline_tests
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install requirements

Install the required Python packages using the provided `requirements.txt`:

```sh
pip install -r requirements.txt
```

**Note:** After installing the base requirements, you also need to install PyTorch according to your operating system:

#### For Windows:
```cmd
pip install -r requirements-torch_windows.txt
```

#### For macOS:
```bash
pip install -r requirements-torch_mac.txt
```

### 3. Create a `.env` file

You need to create a `.env` file in the project root or in the appropriate location for your test setup. This file should contain the necessary environment variables for InfluxDB access. Example:

```
INFLUX_URL=https://your-influxdb-hostname:8086
INFLUX_TOKEN=your-influxdb-token
INFLUX_ORG=your-org
INFLUX_BUCKET=your-bucket
```

**Important:**
- The value for `INFLUX_URL` must be a valid hostname (e.g. `https://influxdb.example.com:8086`), **not an IP address**. The SSL certificate of the server must match this hostname, otherwise certificate verification will fail.

### 4. Run the Complete Pipeline (Recommended)

The easiest way to run all tests is through the unified pipeline:

#### Complete LSTM Weather Pipeline
```sh
python lstm_pipeline_entrypoint.py
```

This pipeline automatically runs all steps in the correct sequence:
1. **InfluxDB Data Query** - Retrieves raw data from InfluxDB
2. **DWD Weather Data Download** - Downloads weather data from German Weather Service
3. **Data Validation** - Validates and cleans the data with automatic scaling factor integration
4. **Weather Data Integration** - Merges weather and PV data
5. **LSTM Model Training** - Trains the model with validated data
6. **Forecast Evaluation** - Evaluates model performance

#### Individual Step Scripts (Alternative)

If you prefer to run individual steps:

##### Step 1: InfluxDB Data Query
```sh
python influxdb_data_query_step.py
```

##### Step 2: DWD Weather Data Download
```sh
python dwd_weatherData/dwd_weatherData_integration_step.py
```

##### Step 3: Data Validation
```sh
python data_validator_step.py
```

##### Step 4: Weather Data Integration
```sh
python dwd_weatherData/merge_dwd_pv_data_step.py
```

##### Step 5: LSTM Model Training
```sh
python lstm_model_training_step.py
```

##### Step 6: Forecast Evaluation
```sh
python forecast_evaluator/forecast_evaluator_step.py
```

### 5. DWD Irr Comparison Analysis (Optional)

For advanced data quality analysis and scaling factor generation:

```sh
python dwd_weatherData/dwd_irr_comparison_analysis.py
```

This analysis:
- Compares DWD weather data with InfluxDB irradiance data
- Generates scaling factors for data validation
- Creates comprehensive visualizations and reports
- Automatically integrates with the DataValidator

**Note:** The scaling factors are automatically used by the DataValidator to improve data quality and fill missing values.

## Configuration

Each test uses its own JSON configuration file:
- `test_config.json` - InfluxDB Test
- `training_test_config.json` - Training Test
- `forecast_test_config.json` - Prediction Test

If these files don't exist, default configurations will be created.

## Results

Test results are saved in the `results/` directory:
- `test_summary_YYYYMMDD_HHMMSS.json` - Summary of all tests
- `detailed_results_YYYYMMDD_HHMMSS.json` - Detailed results
- Subdirectories for specific test results

## Integration with Main System

### Docker Environment (Unchanged)
- All existing functionality continues to work
- No changes to API endpoints or behavior
- Models saved to `models/` directory
- Full SSE support for real-time updates

### Offline Environment (New)
- Uses same codebase as Docker environment
- Automatic fallback to sample data
- Models saved to `offline_models/` directory
- Console-based progress reporting

### Switching Between Modes
**Automatic Detection:**
- If no environment variables are set, the system detects the mode automatically
- Docker containers are automatically detected
- Local development defaults to offline mode



## Important Scripts and Tools

### Main Pipeline Script
- **`lstm_pipeline_entrypoint.py`** - Complete automated LSTM weather pipeline (recommended entry point)

### Core Step Scripts
- **`influxdb_data_query_step.py`** - InfluxDB data retrieval and preprocessing
- **`data_validator_step.py`** - Data validation and quality assurance with scaling factor integration
- **`lstm_model_training_step.py`** - LSTM model training and evaluation
- **`forecast_evaluator/forecast_evaluator_step.py`** - Forecast evaluation and analysis

### Weather Data Integration
- **`dwd_weatherData/dwd_weatherData_integration_step.py`** - German Weather Service (DWD) data download
- **`dwd_weatherData/merge_dwd_pv_data_step.py`** - Weather and PV data integration
- **`dwd_weatherData/dwd_irr_comparison_analysis.py`** - DWD vs InfluxDB irradiance comparison and scaling factor generation







### Configuration Files
- **`results/model_configs/test_lstm_model_settings.json`** - Main model configuration
- **`results/parameter_analysis/lstm_model_settings.json`** - Parameter analysis configuration
- **`results/parameter_analysis/lstm_parameter_settings.json`** - Parameter optimization settings

### Setup and Dependencies
- **`activate_venv.sh`** - Virtual environment activation script
- **`requirements.txt`** - Python dependencies
- **`requirements-torch_mac.txt`** - PyTorch dependencies for macOS
- **`requirements-torch_windows.txt`** - PyTorch dependencies for Windows

## Advanced Analysis Tools

### DWD Weather Data Integration
The `dwd_weatherData/` directory contains comprehensive weather data integration tools:
- **DWD Data Download** - Automated German Weather Service data retrieval
- **Weather-PV Data Merging** - Intelligent merging of weather and PV data
- **Irradiance Comparison Analysis** - DWD vs InfluxDB irradiance comparison
- **Scaling Factor Generation** - Automatic scaling factors for data validation
- **Data Quality Assessment** - Comprehensive weather data quality analysis

See [dwd_weatherData/README_dwd_irr_comparison.md](dwd_weatherData/README_dwd_irr_comparison.md) for detailed documentation.

### Forecast Evaluator
The `forecast_evaluator/` directory contains a comprehensive evaluation framework:
- **Performance Metrics** - MAE, RMSE, MAPE, R², and specialized time series metrics
- **Comparative Analysis** - Model comparison and technology assessment
- **Statistical Analysis** - Bootstrap analysis and significance testing
- **Visualization** - Interactive and static visualization capabilities

See [forecast_evaluator/README.md](forecast_evaluator/README.md) for detailed documentation.

### Experimental Scenarios
The `experimental_scenarios/` directory contains:
- **Model Configurations** - Various model settings and configurations
- **Test Scenarios** - Different experimental setups and parameters
- **Analysis Scripts** - Specialized analysis for specific scenarios

## Notes

- The virtual environment is stored in the `venv/` folder
- All dependencies are defined in `requirements.txt`
- The environment supports both online and offline modes for testing
- The system automatically handles large data queries with chunked retrieval
- Extended timeouts (60 minutes) are configured for large InfluxDB queries