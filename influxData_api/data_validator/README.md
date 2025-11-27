# Data Validator Module

Comprehensive data validation and quality assurance system for PV forecasting data. This module provides multi-layer validation, data cleaning, and quality control for time series data from various sources.

## Overview

The Data Validator is a sophisticated system designed to ensure high-quality data for ML model training and forecasting. It implements scientifically-based validation methods and provides comprehensive data quality reporting.

## Features

### Multi-Layer Validation
- **Range Validation** - Physical limits validation for PV systems
- **Statistical Validation** - Outlier detection using advanced statistical methods
- **Temporal Validation** - Time series consistency and continuity checks
- **Cross-Validation** - Validation across multiple data sources and sensors

### Data Quality Assurance
- **Duplicate Detection** - Identification and handling of duplicate entries
- **Missing Data Handling** - Interpolation and gap-filling strategies
- **Data Consistency** - Cross-sensor validation and consistency checks
- **Quality Scoring** - Comprehensive data quality assessment and scoring

### Advanced Analytics
- **Module Failure Detection** - Automated detection of faulty PV modules
- **Performance Degradation Analysis** - Long-term performance trend analysis
- **Anomaly Detection** - Machine learning-based anomaly identification
- **Data Completeness Analysis** - Temporal and spatial data completeness assessment

## Validation Components

### 1. Range Validation (`range_validator.py`)
Validates data against realistic physical limits for PV systems.

**Validation Rules:**
- Temperature: -40°C to 100°C (module), -40°C to 60°C (ambient)
- Humidity: 0% to 100%
- Irradiance: 0 to 1200 W/m²
- Voltage: 0 to 1000V
- Current: 0 to 20A
- Power: 0 to 500W

### 2. Outlier Detection (`outlier_detector.py`)
Statistical outlier identification using multiple methods.

**Methods:**
- Z-score method with configurable threshold (default: 3.0 standard deviations)
- Modified Z-score method for robust statistics
- Isolation Forest for complex outlier patterns
- Local Outlier Factor (LOF) for density-based detection

### 3. Module Failure Detection (`module_failure_detector.py`)
Automated detection of failing PV modules through relative analysis.

**Detection Methods:**
- Relative performance comparison between modules
- Sustained deviation analysis over time
- Correlation-based failure detection
- Performance degradation trend analysis

### 4. Duplicate Handler (`duplicate_handler.py`)
Comprehensive duplicate detection and handling.

**Features:**
- Temporal duplicate detection
- Sensor-specific duplicate identification
- Aggregation strategies for duplicate data
- Preservation of data integrity

### 5. Data Completeness Analyzer (`completeness_analyzer.py`)
Analysis of data completeness and coverage.

**Metrics:**
- Temporal completeness percentage
- Spatial completeness across modules
- Gap analysis and identification
- Completeness trend analysis

## Usage

### Basic Validation
```python
from data_validator import DataValidator

# Initialize validator
validator = DataValidator(
    config_file="validation_config.json",
    output_dir="results/validation"
)

# Validate data
results = validator.validate_data("input_data.csv")

# Get validation report
report = validator.get_validation_report()
```

### Advanced Validation
```python
# Configure validation parameters
validation_config = {
    "range_validation": {
        "enabled": True,
        "strict_mode": False
    },
    "outlier_detection": {
        "method": "z_score",
        "threshold": 3.0,
        "robust": True
    },
    "module_failure_detection": {
        "enabled": True,
        "correlation_threshold": 0.3,
        "deviation_threshold": 3.0
    }
}

# Run validation with custom config
results = validator.validate_data(
    "input_data.csv",
    config=validation_config
)
```

### Batch Validation
```python
# Validate multiple files
file_list = [
    "data_2024_01.csv",
    "data_2024_02.csv",
    "data_2024_03.csv"
]

batch_results = validator.validate_batch(file_list)
```

## Configuration

### Validation Configuration
```json
{
    "validation_settings": {
        "range_validation": {
            "enabled": true,
            "strict_mode": false,
            "custom_ranges": {
                "temperature": {"min": -40, "max": 100},
                "irradiance": {"min": 0, "max": 1200}
            }
        },
        "outlier_detection": {
            "enabled": true,
            "method": "z_score",
            "threshold": 3.0,
            "robust_statistics": true,
            "window_size": 24
        },
        "module_failure_detection": {
            "enabled": true,
            "correlation_threshold": 0.3,
            "deviation_threshold": 3.0,
            "sustained_deviation_hours": 6
        },
        "duplicate_handling": {
            "enabled": true,
            "aggregation_method": "mean",
            "preserve_metadata": true
        }
    }
}
```

### Quality Thresholds
```json
{
    "quality_thresholds": {
        "excellent": 0.95,
        "good": 0.85,
        "acceptable": 0.70,
        "poor": 0.50
    },
    "completeness_thresholds": {
        "temporal": 0.90,
        "spatial": 0.80,
        "sensor": 0.95
    }
}
```

## Validation Pipeline

### 1. Data Preprocessing
```python
def preprocess_data(raw_data):
    """Preprocess data for validation"""
    # Standardize column names
    standardized_data = standardize_columns(raw_data)
    
    # Convert data types
    converted_data = convert_data_types(standardized_data)
    
    # Handle missing values
    cleaned_data = handle_missing_values(converted_data)
    
    return cleaned_data
```

### 2. Multi-Layer Validation
```python
def validate_data_layers(data):
    """Run multi-layer validation"""
    results = {}
    
    # Layer 1: Range validation
    results['range'] = range_validator.validate(data)
    
    # Layer 2: Outlier detection
    results['outliers'] = outlier_detector.detect(data)
    
    # Layer 3: Module failure detection
    results['failures'] = module_failure_detector.detect(data)
    
    # Layer 4: Duplicate handling
    results['duplicates'] = duplicate_handler.handle(data)
    
    return results
```

### 3. Quality Assessment
```python
def assess_data_quality(validation_results):
    """Assess overall data quality"""
    quality_score = calculate_quality_score(validation_results)
    
    quality_assessment = {
        'score': quality_score,
        'grade': assign_quality_grade(quality_score),
        'recommendations': generate_recommendations(validation_results),
        'actions_required': identify_actions(validation_results)
    }
    
    return quality_assessment
```

## Output Structure

### Validation Results
```
validation_results/
├── validation_run_YYYYMMDD_HHMMSS/
│   ├── 00_duplicates_debug_*.csv          # Duplicate entries analysis
│   ├── 00_duplicates_summary_*.csv        # Duplicate patterns summary
│   ├── 01_range_violations_*.csv          # Data outside physical limits
│   ├── 02_outliers_*.csv                  # Statistical outliers
│   ├── 03_module_failures_*.csv           # Module failure instances
│   ├── 04_validation_summary_*.csv        # Comprehensive statistics
│   ├── 05_FINAL_CLEANED_DATA_*.csv        # Final validated dataset
│   ├── quality_assessment.json            # Quality scoring and recommendations
│   ├── validation_report.html             # Comprehensive HTML report
│   └── validation_plots/                  # Visualization outputs
│       ├── data_quality_overview.png
│       ├── outlier_distribution.png
│       ├── module_performance.png
│       └── temporal_completeness.png
```

### Quality Reports
- **HTML Validation Report** - Comprehensive validation results
- **Quality Assessment JSON** - Detailed quality scoring
- **Cleaned Dataset** - Final validated data ready for ML training
- **Visualization Gallery** - Quality assessment plots and charts


## Integration with Container Environment

### Docker Integration
The Data Validator is designed to work seamlessly in containerized environments:

```dockerfile
# Data Validator Container
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY data_validator/ ./data_validator/
COPY validation_config.json .

CMD ["python", "-m", "data_validator.main"]
```

### Container Configuration
```yaml
# docker-compose.yml
services:
  data_validator:
    build: ./influxData_api/data_validator
    volumes:
      - ./data:/app/data
      - ./results:/app/results
    environment:
      - VALIDATION_CONFIG=/app/validation_config.json
      - OUTPUT_DIR=/app/results
    command: python -m data_validator.main --config validation_config.json
```

### API Integration
```python
# FastAPI endpoint for data validation
@app.post("/validate-data")
async def validate_data_endpoint(request: ValidationRequest):
    validator = DataValidator()
    results = await validator.validate_async(request.data)
    return ValidationResponse(results=results)
```