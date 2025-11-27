# Forecast Evaluator

Comprehensive evaluation framework for ML forecasting models. This module provides detailed performance analysis, statistical evaluation, and comparative assessment of different forecasting models and approaches.

## Overview

The Forecast Evaluator is a sophisticated analysis system designed to provide comprehensive evaluation of PV forecasting models. It includes multiple analytical packages for different aspects of model performance assessment.

## Features

### Performance Metrics
- **Statistical Metrics** - MAE, RMSE, MAPE, R², and other standard metrics
- **Time Series Metrics** - Specialized metrics for time series forecasting
- **Error Analysis** - Detailed error distribution and pattern analysis
- **Confidence Intervals** - Statistical confidence bounds for predictions

### Comparative Analysis
- **Model Comparison** - Side-by-side comparison of different models
- **Technology Comparison** - Silicon vs Perovskite module performance analysis
- **Temporal Analysis** - Performance variation over time and seasons
- **Scenario Analysis** - Performance under different conditions

### Visualization
- **Interactive Plots** - Plotly-based interactive visualizations
- **Static Charts** - Matplotlib-based publication-ready plots
- **Dashboard Views** - Comprehensive dashboard for evaluation results
- **Export Capabilities** - PDF, PNG, and HTML report generation

## Analytical Packages

### Package 1: Basic Performance Analysis
**File**: `package1_basic_performance.py`

Basic performance metrics and statistical evaluation:
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- Mean Absolute Percentage Error (MAPE)
- Coefficient of Determination (R²)
- Mean Bias Error (MBE)

### Package 2: Temporal Performance Analysis
**File**: `package2_temporal_analysis.py`

Time-based performance analysis:
- Hourly performance patterns
- Daily performance trends
- Seasonal performance variations
- Performance stability over time

### Package 3: Intermittency Analysis
**File**: `package3_intermittency_analysis.py`

Analysis of intermittent generation patterns:
- Ramp rate analysis
- Intermittency index calculation
- Power curve analysis
- Generation variability assessment

### Package 4: Scatter Region Analysis
**File**: `package4_scatter_region_analysis.py`

Scatter plot and correlation analysis:
- Prediction vs actual scatter plots
- Correlation coefficient analysis
- Error distribution visualization
- Performance region mapping

## Usage

### Basic Evaluation
```python
from forecast_evaluator import ForecastEvaluator

# Initialize evaluator
evaluator = ForecastEvaluator(
    actual_data="path/to/actual_data.csv",
    predicted_data="path/to/predicted_data.csv",
    output_dir="results/evaluation"
)

# Run comprehensive evaluation
results = evaluator.run_complete_evaluation()
```

### Specific Package Analysis
```python
# Run specific analytical package
package1_results = evaluator.run_package1_basic_performance()
package2_results = evaluator.run_package2_temporal_analysis()
package3_results = evaluator.run_package3_intermittency_analysis()
package4_results = evaluator.run_package4_scatter_region_analysis()
```

### Model Comparison
```python
# Compare multiple models
comparison_results = evaluator.compare_models([
    "lstm_model_results.csv",
    "xgboost_model_results.csv",
    "persistence_model_results.csv"
])
```

## Configuration

### Evaluation Configuration
```json
{
    "evaluation_settings": {
        "metrics": ["MAE", "RMSE", "MAPE", "R2", "MBE"],
        "confidence_level": 0.95,
        "bootstrap_samples": 1000,
        "time_windows": ["hourly", "daily", "monthly"],
        "visualization": {
            "interactive": true,
            "static": true,
            "export_formats": ["png", "pdf", "html"]
        }
    }
}
```

### Analysis Parameters
```json
{
    "analysis_parameters": {
        "intermittency_threshold": 0.1,
        "ramp_rate_window": 15,
        "correlation_method": "pearson",
        "error_bins": 20,
        "confidence_intervals": [0.95, 0.99]
    }
}
```

## Output Structure

### Evaluation Results
```
results/forecast_evaluation/
├── evaluation_YYYYMMDD_HHMMSS/
│   ├── summary_report.html              # Comprehensive HTML report
│   ├── performance_metrics.csv          # Detailed metrics table
│   ├── error_analysis.csv               # Error distribution analysis
│   ├── temporal_analysis.csv            # Time-based performance
│   ├── scatter_analysis.csv             # Correlation and scatter data
│   ├── visualizations/                  # All generated plots
│   │   ├── performance_comparison.png
│   │   ├── error_distribution.png
│   │   ├── temporal_performance.png
│   │   └── scatter_plots.png
│   └── interactive_dashboard.html       # Interactive dashboard
```

### Generated Reports
- **HTML Summary Report** - Comprehensive evaluation report
- **Performance Metrics CSV** - Detailed numerical results
- **Visualization Gallery** - All generated plots and charts
- **Interactive Dashboard** - Web-based exploration interface

## Advanced Features

### Bootstrap Analysis
```python
# Statistical significance testing
bootstrap_results = evaluator.bootstrap_analysis(
    n_samples=1000,
    confidence_level=0.95
)
```

### Error Pattern Analysis
```python
# Identify systematic errors
error_patterns = evaluator.analyze_error_patterns(
    error_types=["bias", "variance", "systematic"]
)
```

### Performance Decomposition
```python
# Decompose performance by factors
decomposition = evaluator.decompose_performance(
    factors=["weather_conditions", "time_of_day", "season"]
)
```

## Integration with Other Components

### Model Training Integration
- Automatic evaluation after model training
- Performance tracking during training
- Model selection based on evaluation results

### Offline Testing Integration
- Integrated with offline testing pipeline
- Automated evaluation workflows
- Batch evaluation of multiple models

### Dashboard Integration
- Results visualization in Streamlit dashboard
- Real-time performance monitoring
- Interactive result exploration

## Performance Metrics Details

### Statistical Metrics
```python
def calculate_mae(actual, predicted):
    """Mean Absolute Error"""
    return np.mean(np.abs(actual - predicted))

def calculate_rmse(actual, predicted):
    """Root Mean Square Error"""
    return np.sqrt(np.mean((actual - predicted) ** 2))

def calculate_mape(actual, predicted):
    """Mean Absolute Percentage Error"""
    return np.mean(np.abs((actual - predicted) / actual)) * 100

def calculate_r2(actual, predicted):
    """Coefficient of Determination"""
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    return 1 - (ss_res / ss_tot)
```

### Time Series Specific Metrics
```python
def calculate_skill_score(actual, predicted, reference):
    """Forecast skill score"""
    mse_model = np.mean((actual - predicted) ** 2)
    mse_reference = np.mean((actual - reference) ** 2)
    return 1 - (mse_model / mse_reference)

def calculate_intermittency_index(power_data):
    """Intermittency index for PV generation"""
    ramps = np.diff(power_data)
    return np.std(ramps) / np.mean(power_data)
```

## Visualization Capabilities

### Interactive Plots
- **Plotly-based visualizations** for web exploration
- **Zoom and pan capabilities** for detailed analysis
- **Hover information** for data point details
- **Export functionality** for presentations

### Static Charts
- **Publication-ready plots** in PNG/PDF format
- **High-resolution graphics** for print media
- **Customizable styling** and color schemes
- **Batch generation** for multiple models

### Dashboard Views
- **Real-time performance monitoring**
- **Comparative model analysis**
- **Trend visualization** over time
- **Interactive filtering** and selection

## Troubleshooting

### Common Issues

1. **Data Format Errors**
   - Ensure consistent timestamp formats
   - Verify data column names and types
   - Check for missing values and outliers

2. **Memory Issues**
   - Use data sampling for large datasets
   - Implement chunked processing
   - Monitor memory usage during evaluation

3. **Visualization Errors**
   - Check data ranges and scales
   - Verify plot configuration parameters
   - Ensure sufficient disk space for outputs

### Debugging
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Validate input data
evaluator.validate_input_data()

# Check evaluation parameters
evaluator.check_evaluation_parameters()

# Test individual packages
evaluator.test_package_execution()
```

## Status

**Current Status**: Fully implemented and functional
- Complete analytical package suite
- Comprehensive performance metrics
- Advanced visualization capabilities
- Statistical analysis and testing
- Integration with offline testing framework

The Forecast Evaluator provides a complete evaluation framework for PV forecasting models with advanced analytical capabilities and comprehensive reporting features.
