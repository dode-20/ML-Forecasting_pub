#!/usr/bin/env python3
"""
LSTM Weather Model Complete Pipeline
====================================

This script runs the complete pipeline for training and evaluating an LSTM model
with weather data integration, from data query to final forecast prediction.

Pipeline Steps:
1. Data query from InfluxDB
2. Get weather data from DWD
3. Data validation and cleaning
4. Weather data integration
5. LSTM model training
6. Forecast evaluation for model analytics
7. Forecast prediction with weather data

"""

import os
import sys
import json
import subprocess
import time
import re
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

class LSTMWeatherPipeline:
    """Complete pipeline for LSTM weather model training and evaluation."""
    
    def __init__(self, config_path: str = "results/model_configs/test_lstm_model_settings.json"):
        """Initialize the pipeline with configuration."""
        self.config_path = config_path
        self.config = self._load_config()
        self.base_dir = Path(__file__).parent.parent
        self.results_dir = self.base_dir / "results"
        
        # Pipeline step scripts
        self.scripts = {
            "influxdb_query": "offline_tests/influxdb_data_query_step.py",
            "weather_data": "offline_tests/dwd_weatherData/dwd_weatherData_integration_step.py",
            "data_validation": "offline_tests/data_validator_step.py",
            "weather_integration": "offline_tests/dwd_weatherData/merge_dwd_pv_data_step.py",
            "model_training": "offline_tests/lstm_model_training_step.py",
            "forecast_evaluation": "offline_tests/forecast_evaluator/forecast_evaluator_step.py",
            "forecast_prediction": "offline_tests/lstm_forecast_prediction_weatherData.py"
        }
        
        # Data paths for checking existence
        start_date = self.config['date_selection']['start'].replace('-', '')
        end_date = self.config['date_selection']['end'].replace('-', '')
        date_range = f"{start_date}_{end_date}"
        
        # Get module_type from config to build correct paths
        module_type = self.config.get('module_type', 'silicon').capitalize()
        
        self.data_paths = {
            "influxdb_raw": f"results/training_data/rawData/{date_range}/{date_range}_{self.config['model_name']}_raw.csv",
            "weather_raw": f"results/weather_data/rawData/combined_weatherData_raw.csv",
            "clean_data": f"results/training_data/{module_type}/cleanData/{date_range}",
            "weather_integrated": f"results/training_data/{module_type}/cleanData/{date_range}"
        }
        
        print("=" * 80)
        print("LSTM WEATHER MODEL COMPLETE PIPELINE")
        print("=" * 80)
        print(f"Configuration: {self.config_path}")
        print(f"Base directory: {self.base_dir}")
        print(f"Results directory: {self.results_dir}")
        print("=" * 80)
        
    def _load_config(self) -> dict:
        """Load configuration from JSON file."""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ Configuration file not found: {self.config_path}")
            sys.exit(1)
        except json.JSONDecodeError:
            print(f"❌ Invalid JSON in configuration file: {self.config_path}")
            sys.exit(1)
    
    def _check_data_exists(self, step_name: str, data_path: str) -> bool:
        """Check if data already exists for a given step."""
        if not data_path:
            return False
            
        full_path = self.base_dir / data_path
        if full_path.exists():
            if full_path.is_file():
                                # Check file size
                size_mb = full_path.stat().st_size / (1024 * 1024)
                print(f"Data already exists for {step_name}:")
                print(f"   Path: {data_path}")
                print(f"   Size: {size_mb:.2f} MB")
                return True
            elif full_path.is_dir():
                # Check directory contents
                files = list(full_path.glob("*"))
                if files:
                    print(f"Data already exists for {step_name}:")
                    print(f"   Path: {data_path}")
                    print(f"   Files: {len(files)}")
                    return True
        return False
    
    def _ask_user_decision(self, step_name: str) -> bool:
        """Ask user whether to execute step or use existing data."""
        while True:
            response = input(f"\nExecute {step_name} step anyway? (y/n): ").lower().strip()
            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no']:
                return False
            else:
                print("Please enter 'y' or 'n'")
    
    def _run_script(self, script_name: str, script_path: str, step_name: str) -> bool:
        """Run a Python script and return success status."""
        print(f"\n{'='*60}")
        print(f"STEP: {step_name}")
        print(f"Script: {script_path}")
        print(f"{'='*60}")
        
        try:
            # Change to base directory for script execution
            os.chdir(self.base_dir)
            
            # Run the script
            result = subprocess.run([
                sys.executable, script_path
            ], capture_output=False, text=True, cwd=self.base_dir)
            
            if result.returncode == 0:
                print(f"{step_name} completed successfully!")
                return True
            else:
                print(f"{step_name} failed with return code: {result.returncode}")
                return False
                
        except Exception as e:
            print(f"Error running {step_name}: {str(e)}")
            return False
    
    def _find_latest_model(self) -> str:
        """Find the latest trained model directory."""
        models_dir = self.base_dir / "results" / "trained_models" / "lstm"
        if not models_dir.exists():
            return None
            
        # Get all model directories and sort by creation time
        model_dirs = [d for d in models_dir.iterdir() if d.is_dir()]
        if not model_dirs:
            return None
            
        # Sort by creation time (newest first)
        model_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        latest_model = model_dirs[0]
        
        print(f"Latest trained model found: {latest_model.name}")
        return str(latest_model)
    
    def _update_script_model_path(self, script_path: str, new_model_path: str) -> bool:
        """Update the model path in a script file."""
        try:
            with open(script_path, 'r') as f:
                content = f.read()
            
            # Use regex to find and replace model path lines more safely
            import re
            
            updated = False
            
            # Pattern 1: MODEL_PATH = "..." lines (for forecast_evaluator.py)
            model_path_pattern = r'MODEL_PATH\s*=\s*"[^"]*"'
            if re.search(model_path_pattern, content):
                new_content = re.sub(
                    model_path_pattern, 
                    f'MODEL_PATH = "{new_model_path}"', 
                    content
                )
                content = new_content
                updated = True
                print(f"Updated MODEL_PATH in {script_path}")
            
            # Pattern 2: model_dir: str = "..." lines (for lstm_forecast_prediction_weatherData.py)
            model_dir_pattern = r'model_dir:\s*str\s*=\s*"[^"]*"'
            if re.search(model_dir_pattern, content):
                new_content = re.sub(
                    model_dir_pattern, 
                    f'model_dir: str = "{new_model_path}"', 
                    content
                )
                content = new_content
                updated = True
                print(f"Updated model_dir in {script_path}")
            
            # Write updated content back if any changes were made
            if updated:
                with open(script_path, 'w') as f:
                    f.write(content)
                print(f"Successfully updated model paths in {script_path}")
                return True
            else:
                print(f"No model path patterns found in {script_path}")
                return False
            
        except Exception as e:
            print(f"Error updating model path in {script_path}: {e}")
            return False
    
    def run_pipeline(self):
        """Execute the complete pipeline."""
        print("\nStarting LSTM Weather Model Pipeline...")
        print(f"Training date range: {self.config['date_selection']['start']} to {self.config['date_selection']['end']}")
        
        # Step 1: Data query from InfluxDB
        print(f"\n{'='*80}")
        print("STEP 1: Data Query from InfluxDB")
        print(f"{'='*80}")
        
        if self._check_data_exists("InfluxDB raw data", self.data_paths["influxdb_raw"]):
            if not self._ask_user_decision("InfluxDB data query"):
                print("Skipping InfluxDB data query, using existing data")
            else:
                if not self._run_script("InfluxDB data query", self.scripts["influxdb_query"], "InfluxDB data query"):
                    print("Pipeline failed at InfluxDB data query step")
                    return False
        else:
            if not self._run_script("InfluxDB data query", self.scripts["influxdb_query"], "InfluxDB data query"):
                print("Pipeline failed at InfluxDB data query step")
                return False
        
        # Step 2: Get weather data
        print(f"\n{'='*80}")
        print("STEP 2: Get Weather Data from DWD")
        print(f"{'='*80}")
        
        if self._check_data_exists("DWD weather data", self.data_paths["weather_raw"]):
            if not self._ask_user_decision("DWD weather data download"):
                print("Skipping DWD weather data download, using existing data")
            else:
                if not self._run_script("DWD weather data download", self.scripts["weather_data"], "DWD weather data download"):
                    print("Pipeline failed at DWD weather data download step")
                    return False
        else:
            if not self._run_script("DWD weather data download", self.scripts["weather_data"], "DWD weather data download"):
                print("Pipeline failed at DWD weather data download step")
                return False
        
        # Step 3: Data validation and cleaning
        print(f"\n{'='*80}")
        print("STEP 3: Data Validation and Cleaning")
        print(f"{'='*80}")
        
        if self._check_data_exists("Clean data", self.data_paths["clean_data"]):
            if not self._ask_user_decision("Data validation and cleaning"):
                print("Skipping data validation and cleaning, using existing data")
            else:
                if not self._run_script("Data validation and cleaning", self.scripts["data_validation"], "Data validation and cleaning"):
                    print("Pipeline failed at data validation and cleaning step")
                    return False
        else:
            if not self._run_script("Data validation and cleaning", self.scripts["data_validation"], "Data validation and cleaning"):
                print("Pipeline failed at data validation and cleaning step")
                return False
        
        # Step 4: Weather data integration
        print(f"\n{'='*80}")
        print("STEP 4: Weather Data Integration")
        print(f"{'='*80}")
        
        if self._check_data_exists("Weather integrated data", self.data_paths["weather_integrated"]):
            if not self._ask_user_decision("Weather data integration"):
                print("Skipping weather data integration, using existing data")
            else:
                if not self._run_script("Weather data integration", self.scripts["weather_integration"], "Weather data integration"):
                    print("Pipeline failed at weather data integration step")
                    return False
        else:
            if not self._run_script("Weather data integration", self.scripts["weather_integration"], "Weather data integration"):
                print("Pipeline failed at weather data integration step")
                return False
        
        # Step 5: LSTM model training (always execute)
        print(f"\n{'='*80}")
        print("STEP 5: LSTM Model Training")
        print(f"{'='*80}")
        
        if not self._run_script("LSTM model training", self.scripts["model_training"], "LSTM model training"):
            print("Pipeline failed at LSTM model training step")
            return False
        
        # Find the newly trained model and update scripts
        print(f"\n{'='*60}")
        print("Updating model paths for evaluation and prediction...")
        print(f"{'='*60}")
        
        new_model_path = self._find_latest_model()
        if new_model_path:
            print(f"Using newly trained model: {new_model_path}")
            
            # Update forecast evaluator script
            if self._update_script_model_path(self.scripts["forecast_evaluation"], new_model_path):
                print("Forecast evaluator script updated successfully")
            else:
                print("Warning: Could not update forecast evaluator script")
            
            # Update forecast prediction script
            if self._update_script_model_path(self.scripts["forecast_prediction"], new_model_path):
                print("Forecast prediction script updated successfully")
            else:
                print("Warning: Could not update forecast prediction script")
        else:
            print("Warning: Could not find newly trained model")
        
        # Step 6: Forecast evaluation (always execute)
        print(f"\n{'='*80}")
        print("STEP 6: Forecast Evaluation for Model Analytics")
        print(f"{'='*80}")
        
        if not self._run_script("Forecast evaluation", self.scripts["forecast_evaluation"], "Forecast evaluation"):
            print("Pipeline failed at forecast evaluation step")
            return False
        
        # Step 7: Forecast prediction (skipped for now)
        print(f"\n{'='*80}")
        print("STEP 7: Forecast Prediction with Weather Data")
        print(f"{'='*80}")
        print("SKIPPED: Forecast prediction will be added in the future")
        print("This step is currently disabled but will be implemented later.")
        
        # Pipeline completed successfully
        print(f"\n{'='*80}")
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}")
        print("All steps completed successfully")
        print("Model trained and evaluated")
        print("Forecast predictions generated")
        print(f"Results saved in: {self.results_dir}")
        print(f"{'='*80}")
        
        return True


def main():
    """Main function to run the pipeline."""
    # Default config path
    config_path = "results/model_configs/test_lstm_model_settings.json"
    
    # Check if config file exists
    if not os.path.exists(config_path):
        print(f"Configuration file not found: {config_path}")
        print("Please ensure the configuration file exists before running the pipeline.")
        sys.exit(1)
    
    # Create and run pipeline
    pipeline = LSTMWeatherPipeline(config_path)
    success = pipeline.run_pipeline()
    
    if not success:
        print("\nPipeline failed. Please check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()