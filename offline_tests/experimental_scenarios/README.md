# Experimental Scenarios Framework

This framework provides systematic analysis capabilities for LSTM model optimization and cross-technology transferability evaluation.

## Structure

```
experimental_scenarios/
├── core/                           # Core framework components
│   ├── experiment_runner.py        # Base experiment execution
│   ├── config_manager.py          # Configuration management
│   └── evaluation_modes.py        # Silicon-only vs transferability modes
├── model_settings/                 # Hyperparameter analysis
│   ├── src/                       # Source code
│   ├── configs/                   # Parameter configurations
│   ├── trained_models/            # Model artifacts
│   └── reports/                   # Analysis reports
├── feature_analysis/              # Feature combination testing
├── forecast_horizon/              # Horizon and sequence analysis
└── reports/                       # Cross-experiment reporting
    ├── silicon_optimization/      # Si-only analysis
    └── transferability_analysis/  # Si→Pvk analysis
```

## Usage

### Model Settings Analysis
```bash
cd experimental_scenarios/model_settings
python run_hyperparameter_analysis.py --mode silicon_only --parameters all
python run_hyperparameter_analysis.py --mode cross_technology --parameters epochs,batch_size
```

## Evaluation Modes

- **silicon_only**: Optimize models for Silicon performance
- **cross_technology**: Evaluate Si→Pvk transferability
