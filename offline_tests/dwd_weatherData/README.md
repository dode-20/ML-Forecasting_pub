# DWD Weather Data Integration

Integration module for German Weather Service (Deutscher Wetterdienst - DWD) data access, processing, and forecasting. This module provides comprehensive weather data integration for PV forecasting models.

## Overview

The DWD Weather Data Integration module provides access to historical and forecast weather data from the German Weather Service. It supports data retrieval, processing, and integration with PV forecasting models, particularly for weather-informed predictions.

## Features

### Data Access
- **Historical Weather Data** - Access to historical weather measurements
- **Forecast Data** - ICON-D2 numerical weather prediction model data
- **Real-time Data** - Current weather conditions and measurements
- **Multiple Data Sources** - Various DWD data products and services

### Data Processing
- **Data Validation** - Quality checks and validation of weather data
- **Data Cleaning** - Handling of missing values and outliers
- **Data Aggregation** - Time-based aggregation and resampling
- **Format Standardization** - Consistent data format for ML models

### Weather Integration
- **PV Forecasting** - Weather data integration for PV energy forecasting
- **Feature Engineering** - Weather-based feature creation for ML models
- **Forecast Horizon** - Multi-hour and multi-day weather forecasts
- **Irradiance Processing** - Solar irradiance data processing and validation

## Components

### 1. DWD Weather Data Integration (`dwd_weatherData_integration.py`)
Main integration module for DWD data access and processing.

**Features:**
- Historical weather data retrieval
- Data validation and quality control
- Time series processing and aggregation
- Integration with PV forecasting pipeline

### 2. ICON-D2 Forecast (`icon_d2_forecast.py`)
Integration with ICON-D2 numerical weather prediction model.

**Features:**
- Forecast data retrieval
- Multi-hour prediction horizons
- Weather parameter extraction
- Forecast data validation

### 3. Data Merging Utilities
- **Merge DWD Forecast Data** (`merge_dwd_forecast_data.py`) - Combines forecast data with PV data
- **Merge DWD PV Data** (`merge_dwd_pv_data.py`) - Integrates weather data with PV measurements

## Weather Parameters

### Available Weather Variables
- **Temperature** - Air temperature at 2m height (TT_2m)
- **Relative Humidity** - Relative humidity (RF_2m)
- **Wind Speed** - Wind speed at 10m height (RWS_10)
- **Wind Direction** - Wind direction indicator (RWS_IND_10)
- **Visibility** - Horizontal visibility (V_N)
- **Global Solar Radiation** - Global solar radiation (GS_10)
- **Cloud Cover** - Total cloud cover (N)

### Data Resolution
- **Historical Data**: 10-minute, hourly, daily resolution
- **Forecast Data**: 1-hour resolution up to 48 hours ahead
- **Real-time Data**: 10-minute resolution

## Usage

### Basic Data Retrieval
```python
from dwd_weatherData_integration import DWDOptimizedIntegrator

# Initialize integrator
integrator = DWDOptimizedIntegrator(
    station_id="01050",  # Hamburg station
    start_date="2024-01-01",
    end_date="2024-12-31"
)

# Retrieve historical weather data
weather_data = integrator.get_historical_data()

# Retrieve forecast data
forecast_data = integrator.get_forecast_data()
```


### Data Merging
```python
from merge_dwd_pv_data import merge_weather_pv_data

# Merge weather data with PV data
combined_data = merge_weather_pv_data(
    pv_data_path="pv_data.csv",
    weather_data_path="weather_data.csv",
    output_path="combined_data.csv"
)
```

## Configuration

### DWD Station Configuration
```python
DWD_STATIONS = {
    "01050": {  # Hamburg
        "name": "Hamburg",
        "latitude": 53.5511,
        "longitude": 9.9937,
        "elevation": 15.0
    },
    "01028": {  # Berlin
        "name": "Berlin",
        "latitude": 52.5200,
        "longitude": 13.4050,
        "elevation": 34.0
    }
}
```

### Weather Parameters Configuration
```python
WEATHER_PARAMETERS = {
    "temperature": "TT_2m",
    "humidity": "RF_2m",
    "wind_speed": "RWS_10",
    "wind_direction": "RWS_IND_10",
    "visibility": "V_N",
    "global_radiation": "GS_10",
    "cloud_cover": "N"
}
```

### API Configuration
```python
DWD_API_CONFIG = {
    "base_url": "https://opendata.dwd.de",
    "timeout": 30,
    "retry_attempts": 3,
    "retry_delay": 5
}
```

## Data Processing Pipeline

### 1. Data Retrieval
```python
def retrieve_weather_data(station_id, start_date, end_date):
    """Retrieve weather data from DWD"""
    # API request to DWD
    raw_data = dwd_api.get_data(station_id, start_date, end_date)
    return raw_data
```

### 2. Data Validation
```python
def validate_weather_data(data):
    """Validate weather data quality"""
    # Range validation
    validated_data = validate_ranges(data)
    
    # Outlier detection
    cleaned_data = detect_outliers(validated_data)
    
    # Missing value handling
    processed_data = handle_missing_values(cleaned_data)
    
    return processed_data
```

### 3. Data Processing
```python
def process_weather_data(data):
    """Process weather data for ML models"""
    # Time-based aggregation
    aggregated_data = aggregate_by_time(data, "1H")
    
    # Feature engineering
    features = create_weather_features(aggregated_data)
    
    # Format standardization
    standardized_data = standardize_format(features)
    
    return standardized_data
```

## Integration with PV Forecasting

### Weather-Informed Models
```python
# Weather features for LSTM models
weather_features = [
    "TT_10",      # Temperature
    "RF_10",      # Relative humidity
    "RWS_10",     # Wind speed
    "RWS_IND_10", # Wind direction
    "V_N",        # Visibility
    "GS_10"       # Global solar radiation
]

# Multi-step weather-informed prediction
def weather_informed_prediction(historical_data, weather_forecast):
    """Combine historical PV data with weather forecast"""
    # Prepare input sequence with weather data
    input_sequence = prepare_sequence_with_weather(
        historical_data, weather_forecast
    )
    
    # Make prediction
    prediction = model.predict(input_sequence)
    
    return prediction
```

### Forecast Horizon Integration
```python
def integrate_forecast_horizon(pv_data, weather_forecast, horizon_hours=6):
    """Integrate weather forecast for multi-step prediction"""
    # Align weather forecast with prediction horizon
    aligned_forecast = align_forecast_to_horizon(
        weather_forecast, horizon_hours
    )
    
    # Create combined input
    combined_input = combine_pv_weather_data(
        pv_data, aligned_forecast
    )
    
    return combined_input
```



## Status

**Current Status**: Partially implemented
- Basic DWD data integration
- ICON-D2 forecast support
- Data processing utilities
- PV data integration
- Advanced validation (in development)
- Performance optimization (in development)
- Comprehensive error handling (in development)

The module provides a solid foundation for weather data integration but can be further developed to support all features of the comprehensive offline testing framework and provide advanced weather-based forecasting capabilities.
