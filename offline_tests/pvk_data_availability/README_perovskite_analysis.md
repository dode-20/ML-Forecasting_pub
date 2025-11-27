# Perovskite Data Availability Analysis

This directory contains tools for analyzing data availability between Silicon (Si) and Perovskite (Pvk) modules as part of the precondition assessment for forecasting model comparison.

## Scripts

### `perovskite_data_availability_analysis.py`
Main analysis script that performs comprehensive data availability assessment.

**Features:**
- Heatmaps of daily data availability per module
- Aggregated counts of available modules per day for each technology
- Identification of overlapping training, validation, and test windows
- Balanced seasonal coverage analysis
- Interactive HTML reports and static plots
- Comprehensive summary files

### `run_perovskite_precondition_analysis.py`
Example script showing how to run the analysis with typical file paths.

## Usage

### Command Line Usage
```bash
python perovskite_data_availability_analysis.py \
    --silicon-path "path/to/silicon_clean_data.csv" \
    --perovskite-path "path/to/perovskite_clean_data.csv" \
    --output-dir "path/to/output/directory"  # optional
```

### Example Usage
```bash
# Using typical project paths
python perovskite_data_availability_analysis.py \
    --silicon-path "../results/training_data/Silicon/cleanData/20240901_20250725/20240901_20250725_test_lstm_model_clean-1h.csv" \
    --perovskite-path "../results/training_data/Perovskite/cleanData/20240901_20250725/20240901_20250725_test_lstm_model_clean-1h.csv"

# Or run the example script (update paths first)
python run_perovskite_precondition_analysis.py
```

## Input Requirements

### Data Format
Both Silicon and Perovskite CSV files should contain:
- `_time` or `timestamp` column with datetime information
- `Name` column with module names (optional for averaged data)
- Measurement columns (P, U, I, Temp, etc.)

### Typical File Paths
```
results/training_data/
├── Silicon/cleanData/YYYYMMDD_YYYYMMDD/
│   ├── *_silicon_clean-5min.csv
│   ├── *_silicon_clean-10min.csv
│   └── *_silicon_clean-1h.csv
└── Perovskite/cleanData/YYYYMMDD_YYYYMMDD/
    ├── *_perovskite_clean-5min.csv
    ├── *_perovskite_clean-10min.csv
    └── *_perovskite_clean-1h.csv
```

## Output Structure

```
results/perovskite_analysis/preconditions/YYYYMMDD_HHMMSS/
├── precondition_analysis_report.html             # Comprehensive HTML report (MAIN REPORT)
├── summary.txt                                    # Text summary
├── calendar_heatmap_silicon.png                  # Silicon calendar heatmap
├── calendar_heatmap_perovskite.png               # Perovskite calendar heatmap
├── calendar_comparison_heatmap.png               # Side-by-side comparison with overlap
├── temporal_overlap_analysis.png                 # Detailed overlap analysis plots
├── interactive_calendar_silicon.html             # Interactive Silicon calendar
├── interactive_calendar_perovskite.html          # Interactive Perovskite calendar
└── interactive_calendar_comparison.html          # Interactive comparison
```

## Generated Analysis

### 1. Basic Statistics
- Total records per technology
- Date ranges and data resolution
- Number of unique modules
- Missing value analysis

### 2. Calendar-Style Heatmaps
- **Static calendar heatmaps**: Daily timestamp counts in calendar layout (PNG)
- **Interactive calendar heatmaps**: Hoverable calendar views with detailed timestamps (HTML)
- **Combined comparisons**: Side-by-side Si vs Pvk calendar views
- **Color coding**: 0-24 timestamps per day (for 1h resolution data)

### 3. Temporal Overlap Analysis
- Overlapping days between Si and Pvk data
- Technology-specific coverage periods
- Continuous overlap period identification
- Coverage distribution visualization

### 4. Seasonal Coverage Analysis
- Winter, Spring, Summer, Autumn data availability
- Balanced coverage assessment
- Seasonal comparison between technologies

### 5. Recommendations
- Suitability for fair model comparison
- Data quality assessment
- Identified limitations and suggestions

## Analysis Workflow

1. **Data Loading**: Validate and load clean CSV files
2. **Availability Analysis**: 
   - Daily availability matrices
   - Module-level statistics
   - Temporal overlap calculation
   - Seasonal coverage assessment
3. **Visualization Creation**:
   - Static heatmaps (matplotlib/seaborn)
   - Interactive plots (plotly)
   - Overlap analysis charts
4. **Report Generation**:
   - Text summary with recommendations
   - HTML interactive report

## Example Output

The analysis generates:
- **Heatmaps** showing when each module has data available
- **Timeline plots** showing the number of active modules over time
- **Overlap analysis** identifying suitable periods for model comparison
- **Seasonal coverage** ensuring balanced training across seasons
- **Interactive HTML report** for detailed exploration
- **Text summary** with concrete recommendations

## Notes

- Handles both individual module data and pre-averaged technology data
- Automatically detects data resolution (5min, 10min, 1h)
- Creates timestamped output directories to preserve analysis history
- Provides both static (PNG) and interactive (HTML) visualizations
- Generates actionable recommendations for model comparison validity
