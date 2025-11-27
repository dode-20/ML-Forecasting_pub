# DWD GS_10 vs InfluxDB Irr Comparison Analysis

This script compares Global Solar Radiation (GS_10) values from the German Weather Service (DWD) with Irradiance (Irr) values from InfluxDB to calculate scaling factors and analyze deviations.

## Overview

The script performs a scientific analysis to:
- Calculate scaling factors between DWD GS_10 and InfluxDB Irr
- Identify time-dependent patterns (hourly, monthly, seasonal)
- Visualize deviations between the two data sources
- Provide recommendations for using scaling factors

## Prerequisites

- Python 3.7+
- Required Python packages (see `requirements_dwd_irr_comparison.txt`)

## Installation

1. Install the required packages:
```bash
pip install -r requirements_dwd_irr_comparison.txt
```

## Usage

### Automatic Mode (Recommended)

The script automatically detects available data files in the `rawData` directory:

```bash
python dwd_irr_comparison_analysis.py
```

### Manual Data Preparation

To ensure the data validation works correctly with scaling factors, you need to manually prepare the data files:

#### Step 1: Prepare the `rawData` Directory

Create the directory structure:
```bash
mkdir -p results/irr_comparison/rawData
```

#### Step 2: Add DWD Weather Data

Copy the DWD weather data file to the rawData directory:
```bash
cp results/weather_data/rawData/combined_weatherData-10min.csv results/irr_comparison/rawData/
```

#### Step 3: Add Clean InfluxDB Data

Copy a clean InfluxDB data file with the same resolution (10min) to the rawData directory:
```bash
cp results/training_data/Silicon/cleanData/[date_range]/[date_range]_[model_name]_clean-10min.csv results/irr_comparison/rawData/
```

**Example:**
```bash
cp results/training_data/Silicon/cleanData/20240725_20250715/20240725_20250715_test_lstm_model_clean-10min.csv results/irr_comparison/rawData/
```

#### Step 4: Verify Files

Your `results/irr_comparison/rawData/` directory should contain:
- `combined_weatherData-10min.csv` (DWD weather data)
- `[date_range]_[model_name]_clean-10min.csv` (Clean InfluxDB data)

#### Step 5: Run Analysis

```bash
python dwd_irr_comparison_analysis.py
```

### Command Line with Explicit Paths

```bash
python dwd_irr_comparison_analysis.py \
    --dwd-csv "results/irr_comparison/rawData/combined_weatherData-10min.csv" \
    --influx-csv "results/irr_comparison/rawData/your_clean_data.csv" \
    --output-dir "results/irr_comparison"
```

### Parameters

- `--dwd-csv`: Path to DWD weather data CSV file (must contain `timestamp` and `GS_10` columns)
- `--influx-csv`: Path to InfluxDB data CSV file (must contain `Irr` column and timestamp)
- `--output-dir`: Output directory for results (default: "results/irr_comparison")

### Programmatic Usage

```python
from dwd_irr_comparison_analysis import DWDIrrComparisonAnalyzer

# Create analyzer
analyzer = DWDIrrComparisonAnalyzer(
    dwd_csv_path="path/to/dwd_data.csv",
    influx_csv_path="path/to/influx_data.csv",
    output_dir="results"
)

# Run complete analysis
analyzer.run_complete_analysis()

# Or individual steps
analyzer.load_dwd_data()
analyzer.load_influx_data()
analyzer.merge_data()
analyzer.calculate_scaling_factors()
analyzer.create_visualizations()
analyzer.save_results()
analyzer.print_summary_report()
```

## Input Data

### DWD Data
- **Required columns**: `timestamp`, `GS_10`
- **Format**: CSV with timestamp and Global Solar Radiation in W/m²
- **Example**:
```csv
timestamp,GS_10
2024-02-21 01:00:00,0.0
2024-02-21 01:10:00,0.0
2024-02-21 12:00:00,450.2
```

### InfluxDB Data
- **Required columns**: Irr column (automatically detected: `Irr`, `irr`, `IRR`, `irradiance`, `G`) and timestamp
- **Format**: CSV with timestamp and Irradiance in W/m²
- **Example**:
```csv
timestamp,Irr
2024-02-21 01:00:00,0.0
2024-02-21 01:10:00,0.0
2024-02-21 12:00:00,480.5
```

## Outputs

### Files
- `daily_scaling_factors.csv`: Daily scaling factors with statistics (used by DataValidator)
- `hourly_scaling_factors.csv`: Hourly scaling factors with statistics
- `monthly_scaling_factors.csv`: Monthly scaling factors
- `seasonal_scaling_factors.csv`: Seasonal scaling factors
- `overall_statistics.csv`: Overall statistics
- `sample_combined_data.csv`: Sample of merged data
- `data_summary_statistics.csv`: Summary data statistics
- `dwd_irr_comparison_analysis.png`: Comprehensive visualizations

### Visualizations
The panel creates 9 different plots:
1. **Scatter Plot**: GS_10 vs Irr
2. **Histogram**: Distribution of scaling factors
3. **Hourly Factors**: Scaling factors per hour with standard deviation
4. **Monthly Factors**: Scaling factors per month
5. **Time Series**: Scaling factors over time
6. **Box Plot**: Distribution of scaling factors per hour
7. **Seasonal Factors**: Scaling factors per season
8. **Correlation Matrix**: Correlation between GS_10, Irr and scaling factor
9. **Residual Plot**: Prediction residuals

## Scaling Factors

### Calculation
The scaling factor is calculated as:
```
Scaling Factor = Irr (InfluxDB) / GS_10 (DWD)
```

### Interpretation
- **Factor > 1**: InfluxDB Irr is higher than DWD GS_10
- **Factor < 1**: InfluxDB Irr is lower than DWD GS_10
- **Factor = 1**: Perfect match

### Usage in Data Validation
The scaling factors are used by the DataValidator's IrrGapFiller to:
- Fill missing irradiance values using DWD weather data
- Apply appropriate scaling based on time of day and season
- Improve data quality and completeness

The `daily_scaling_factors.csv` file is automatically loaded by the DataValidator from the most recent analysis in `results/irr_comparison/`.

## Data Quality

### Filtering
The script automatically filters:
- Night values (GS_10 = 0, Irr = 0)
- Invalid scaling factors (≤ 0 or > 10)
- Missing values

### Validation
- Timestamp matching on minute level
- Statistical outlier detection
- Correlation analysis

## Integration with DataValidator

### Automatic Usage
The DataValidator automatically uses the scaling factors:
1. Looks for the most recent `daily_scaling_factors.csv` in `results/irr_comparison/`
2. Uses the scaling factors to fill missing irradiance values
3. Applies time-dependent scaling (morning, midday, evening)

### Manual Updates
To update scaling factors:
1. Run the DWD Irr comparison analysis with new data
2. The new scaling factors will be automatically used by the DataValidator
3. No manual configuration required

## Logging

The script creates detailed logs in:
- Console (standard output)
- File: `dwd_irr_comparison.log`

## Error Handling

The script handles various error scenarios:
- Missing or invalid CSV files
- Missing required columns
- Data format issues
- Memory issues with large datasets

## Performance

- **Optimized for**: Datasets up to several million records
- **Memory usage**: Efficient pandas operations
- **Visualization**: Sampling for large datasets

## Examples

### Simple Analysis
```bash
python dwd_irr_comparison_analysis.py
```

### Custom Output Directory
```bash
python dwd_irr_comparison_analysis.py \
    --output-dir "custom_results"
```

### Manual File Specification
```bash
python dwd_irr_comparison_analysis.py \
    --dwd-csv "custom_dwd_data.csv" \
    --influx-csv "custom_influx_data.csv"
```

## Troubleshooting

### Common Issues

1. **Missing Files**: Ensure both DWD and InfluxDB data files are in `results/irr_comparison/rawData/`
2. **Missing Columns**: Check that CSV files contain required columns
3. **Timestamp Format**: Ensure timestamps are in a readable format
4. **Memory Issues**: For very large datasets, memory usage can be high
5. **Matplotlib Errors**: In headless environments, `plt.show()` can cause issues

### Solutions

- Check the log file for detailed error messages
- Ensure all required packages are installed
- Use smaller datasets for testing
- Adjust visualization settings for your environment

### Data Preparation Checklist

Before running the analysis, ensure:
- [ ] `results/irr_comparison/rawData/` directory exists
- [ ] `combined_weatherData-10min.csv` is in the rawData directory
- [ ] Clean InfluxDB data file (10min resolution) is in the rawData directory
- [ ] Both files have matching timestamp formats
- [ ] Required columns are present in both files

## License

This script is part of the ML-Forecasting project.

## Support

For questions or issues, contact the ML-Forecasting team.