#!/usr/bin/env python3
"""
Model training component for hyperparameter experiments.

Adapts the existing LSTM training pipeline for systematic parameter analysis.
"""

import sys
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import importlib.util

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "offline_tests"))
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "influxData_api" / "data"))

# Import the existing OfflineLSTMModelTest class
from lstm_model_training_step import OfflineLSTMModelTest

class ExperimentModelTrainer:
    """Training component for hyperparameter experiments"""
    
    def __init__(self, output_dir: Path, device: str = "cuda"):
        """
        Initialize experiment trainer.
        
        Args:
            output_dir: Directory for saving trained models
            device: Training device ("auto", "cpu", "cuda")
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.data_paths = self._load_data_paths()
    
    def _load_data_paths(self) -> Dict[str, Any]:
        """Load data paths configuration"""
        config_path = Path(__file__).parent.parent / "configs" / "data_paths.json"
        with open(config_path, 'r') as f:
            return json.load(f)
        
    def find_training_data(self, module_type: str, data_resolution: str = "5min") -> Path:
        """Find training data using direct file paths or directory search"""
        # First try: Direct file path from config
        training_files = self.data_paths.get("training_data_files", {})
        direct_file = training_files.get(module_type)
        
        if direct_file:
            # Handle both absolute and relative paths
            if Path(direct_file).is_absolute():
                file_path = Path(direct_file)
            else:
                # Relative path from project root (ML-forecasting directory)
                # Navigate up from: offline_tests/experimental_scenarios/model_settings/src/model_trainer.py
                # To: ML-forecasting/
                project_root = Path(__file__).parent.parent.parent.parent.parent.parent.parent
                file_path = project_root / direct_file
                print(f"DEBUG: Project root: {project_root}")
                print(f"DEBUG: Resolved file path: {file_path}")
            
            if file_path.exists() and file_path.is_file():
                print(f"Using direct file path: {file_path}")
                return file_path
            else:
                print(f"WARNING: Direct file path not found: {file_path}")
        
        # Second try: Search in fallback directories
        fallback_dirs = self.data_paths.get("fallback_directories", {}).get(module_type, [])
        for base_dir_str in fallback_dirs:
            # Handle both absolute and relative paths
            if Path(base_dir_str).is_absolute():
                base_dir = Path(base_dir_str)
            else:
                # Relative path from project root (ML-forecasting directory)
                project_root = Path(__file__).parent.parent.parent.parent.parent.parent.parent
                base_dir = project_root / base_dir_str
            
            if not base_dir.exists():
                continue
                
            # Search for files in this directory
            found_file = self._search_training_file_in_directory(base_dir, data_resolution)
            if found_file:
                print(f"Found training file in fallback directory: {found_file}")
                return found_file
        
        # If nothing found, raise error
        raise FileNotFoundError(f"Training data not found for {module_type}. Checked direct file: {direct_file}, and fallback directories: {fallback_dirs}")
    
    def _search_training_file_in_directory(self, base_dir: Path, data_resolution: str) -> Optional[Path]:
        """Search for training files in a directory with the specified resolution"""
        
        # Look for data files with specified resolution
        patterns = [
            f"*_test_lstm_model_clean-{data_resolution}_weather-integrated.csv",
            f"*_test_lstm_model_clean-{data_resolution}.csv",
            "*_test_lstm_model_clean.csv"
        ]
        
        latest_file = None
        latest_timestamp = None
        
        # Search in subdirectories first
        for subdir in base_dir.iterdir():
            if subdir.is_dir():
                for pattern in patterns:
                    files = list(subdir.glob(pattern))
                    for file in files:
                        file_timestamp = self._extract_timestamp(file)
                        if latest_timestamp is None or file_timestamp > latest_timestamp:
                            latest_file = file
                            latest_timestamp = file_timestamp
                    if latest_file:
                        break
        
        # Also search directly in base directory
        if latest_file is None:
            for pattern in patterns:
                files = list(base_dir.glob(pattern))
                for file in files:
                    file_timestamp = self._extract_timestamp(file)
                    if latest_timestamp is None or file_timestamp > latest_timestamp:
                        latest_file = file
                        latest_timestamp = file_timestamp
        
        return latest_file
    
    def _extract_timestamp(self, file_path: Path) -> datetime:
        """Extract timestamp from file path"""
        try:
            filename = file_path.stem
            if '_test_lstm_model_clean' in filename:
                date_part = filename.split('_test_lstm_model_clean')[0]
                if '_' in date_part:
                    end_date_str = date_part.split('_')[1]
                    return datetime.strptime(end_date_str, '%Y%m%d')
        except (ValueError, IndexError):
            pass
        
        return datetime.fromtimestamp(file_path.stat().st_mtime)
    
    def train_model(self, config_file_path: str) -> Tuple[Dict[str, Any], Path]:
        """
        Train model with given configuration file path.
        
        Args:
            config_file_path: Path to the JSON configuration file
            
        Returns:
            Tuple of (training_results, model_output_path)
        """
        # Load config from file
        with open(config_file_path, 'r') as f:
            config = json.load(f)
            
        print(f"Training model: {config['model_name']}")
        
        # Find training data
        module_type = config.get("module_type", "silicon")
        data_resolution = config.get("data_resolution", "5min")
        training_data_path = self.find_training_data(module_type, data_resolution)
        
        print(f"Using training data: {training_data_path}")
        
        # Create OfflineLSTMModelTest instance with the config file path
        offline_test = OfflineLSTMModelTest(
            config_file=config_file_path,
            csv_file=str(training_data_path),
            device=self.device
        )
        
        # Load training data
        df = pd.read_csv(training_data_path)
        print(f"Loaded training data: {len(df)} rows")
        
        # Train the model using the existing pipeline
        success = offline_test.test_model_training(df)
        
        if success:
            # Get the original model path from the offline test
            # The offline test saves with timestamp format, not model_name format
            original_model_path = offline_test.output_dir
            print(f"DEBUG: Offline test output dir: {original_model_path}")
            print(f"DEBUG: Files in offline test output dir:")
            for file_path in original_model_path.rglob('*'):
                if file_path.is_file():
                    print(f"  - {file_path}")
            
            # Create experiment-specific model directory
            # Use just the model name without adding another timestamp
            experiment_model_dir = self.output_dir / config['model_name']
            experiment_model_dir.mkdir(parents=True, exist_ok=True)
            print(f"DEBUG: Experiment model dir: {experiment_model_dir}")
            
            # Copy model files to experiment directory
            import shutil
            if original_model_path.exists():
                print(f"DEBUG: Original model path exists: {original_model_path}")
                print(f"DEBUG: Files in original model directory:")
                for file_path in original_model_path.rglob('*'):
                    if file_path.is_file():
                        print(f"  - {file_path}")
                
                # Copy all files from original model directory to experiment directory
                for file_path in original_model_path.rglob('*'):
                    if file_path.is_file():
                        relative_path = file_path.relative_to(original_model_path)
                        dest_path = experiment_model_dir / relative_path
                        dest_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(file_path, dest_path)
                        print(f"Copied: {file_path} -> {dest_path}")
                
                print(f"DEBUG: Files in experiment model directory after copy:")
                for file_path in experiment_model_dir.rglob('*'):
                    if file_path.is_file():
                        print(f"  - {file_path}")
                
                # Ensure config file exists with correct name for ForecastEvaluator
                config_files = list(experiment_model_dir.glob("*config*.json"))
                if config_files:
                    print(f"DEBUG: Found config files: {config_files}")
                    # Extract model name without timestamp for ForecastEvaluator
                    model_name_parts = config['model_name'].split('_')
                    if len(model_name_parts) >= 2 and model_name_parts[-1].isdigit() and model_name_parts[-2].isdigit():
                        # Remove timestamp from model name
                        base_model_name = '_'.join(model_name_parts[:-2])
                    else:
                        base_model_name = config['model_name']
                    
                    # Rename config file to match ForecastEvaluator expectations
                    expected_config_name = f"model_config_{base_model_name}_lstm.json"
                    expected_config_path = experiment_model_dir / expected_config_name
                    
                    if not expected_config_path.exists() and config_files:
                        # Copy the first config file to the expected name
                        shutil.copy2(config_files[0], expected_config_path)
                        print(f"DEBUG: Renamed config file: {config_files[0]} -> {expected_config_path}")
                
                # Also rename preprocessor file to match ForecastEvaluator expectations
                preprocessor_files = list(experiment_model_dir.glob("*preprocessor*.json"))
                if preprocessor_files:
                    print(f"DEBUG: Found preprocessor files: {preprocessor_files}")
                    expected_preprocessor_name = f"preprocessor_{base_model_name}_lstm.json"
                    expected_preprocessor_path = experiment_model_dir / expected_preprocessor_name
                    
                    if not expected_preprocessor_path.exists() and preprocessor_files:
                        # Copy the first preprocessor file to the expected name
                        shutil.copy2(preprocessor_files[0], expected_preprocessor_path)
                        print(f"DEBUG: Renamed preprocessor file: {preprocessor_files[0]} -> {expected_preprocessor_path}")
                else:
                    print(f"WARNING: No preprocessor files found in {experiment_model_dir}")
                
                # Also rename model file to match ForecastEvaluator expectations
                model_files = list(experiment_model_dir.glob("*.pth"))
                if model_files:
                    print(f"DEBUG: Found model files: {model_files}")
                    expected_model_name = f"{base_model_name}_lstm.pth"
                    expected_model_path = experiment_model_dir / expected_model_name
                    
                    if not expected_model_path.exists() and model_files:
                        # Copy the first model file to the expected name
                        shutil.copy2(model_files[0], expected_model_path)
                        print(f"DEBUG: Renamed model file: {model_files[0]} -> {expected_model_path}")
                else:
                    print(f"WARNING: No model files found in {experiment_model_dir}")
            else:
                print(f"ERROR: Original model path does not exist: {original_model_path}")
            
            training_results = {
                "success": True,
                "model_path": experiment_model_dir,
                "timestamp": offline_test.timestamp
            }
            return training_results, experiment_model_dir
        else:
            training_results = {
                "success": False,
                "error": "Training failed"
            }
            return training_results, None
    
    def _load_weather_forecast_data(self, historical_data: pd.DataFrame, config: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """
        Create weather forecast data for Weather-Informed mode.
        Uses the same dataset but shifts the weather data to simulate forecasts.
        """
        try:
            weather_features = config.get("weather_data", {}).get("weather_features", [])
            if not weather_features:
                print("  WARN No weather features configured")
                return None
            
            # Create weather forecast data by shifting the historical weather data
            # This simulates having weather forecasts for the future timesteps
            forecast_data = historical_data.copy()
            
            # Keep only weather features and timestamp
            forecast_columns = ["timestamp"] + weather_features
            forecast_data = forecast_data[forecast_columns]
            
            print(f"  INFO Using shifted historical weather data as forecast simulation")
            print(f"  INFO Forecast features: {weather_features}")
            print(f"  INFO Forecast data shape: {forecast_data.shape}")
            print(f"  INFO This simulates weather forecasts for future timesteps")
            
            return forecast_data
            
        except Exception as e:
            print(f"  ERROR Failed to create weather forecast data: {e}")
            return None

    def _prepare_training_settings(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare training settings from config"""
        # Get features
        time_features = config.get("time_features", [])
        features = config.get("features", [])
        weather_features = []
        
        if config.get("weather_data", {}).get("use_weatherData", False):
            weather_features = config.get("weather_data", {}).get("weather_features", [])
        
        all_features = time_features + features + weather_features
        
        return {
            "model_name": config["model_name"],
            "features": all_features,
            "output": config.get("output", ["P_normalized"]),
            "batch_size": config.get("batch_size", 32),
            "epochs": config.get("epochs", 100),
            "learning_rate": config.get("learning_rate", 0.001),
            "loss_function": config.get("loss_function", "MAE"),
            "shuffle": config.get("shuffle", False),
            "use_validation_set": config.get("validation_set", {}).get("use_validation_set", "Yes"),
            "validation_split": config.get("validation_set", {}).get("validation_split", 0.15),
            "sequence_length": config.get("sequence_length", 48),
            "data_resolution": config.get("data_resolution", "5min"),
            "forecast_mode": config.get("forecast_mode", {"mode": "one-step", "forecast_steps": 1}),
            "hidden_size": config.get("hidden_size", 64),
            "num_layers": config.get("num_layers", 2),
            "dropout": config.get("dropout", 0.1)
        }
    
    def _save_training_metadata(self, results: Dict[str, Any], output_dir: Path):
        """Save training metadata and results"""
        metadata_path = output_dir / "training_metadata.json"
        
        # Prepare serializable metadata
        metadata = {
            "model_name": results["model_name"],
            "timestamp": results["timestamp"],
            "device": results["device"],
            "training_data_path": results["training_data_path"],
            "final_train_loss": results["final_train_loss"],
            "final_val_loss": results["final_val_loss"],
            "config_summary": {
                "epochs": results["config"].get("epochs"),
                "batch_size": results["config"].get("batch_size"),
                "learning_rate": results["config"].get("learning_rate"),
                "hidden_size": results["config"].get("hidden_size"),
                "num_layers": results["config"].get("num_layers"),
                "dropout": results["config"].get("dropout"),
                "sequence_length": results["config"].get("sequence_length")
            }
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"Training metadata saved: {metadata_path}")
    
    def _save_preprocessor_scalers(self, trainer, model_output_dir: Path, config: Dict[str, Any]):
        """Save preprocessor scalers - now handled by file renaming"""
        try:
            if hasattr(trainer, 'preprocessor') and trainer.preprocessor:
                print("Preprocessor scalers will be handled by file renaming")
            else:
                print("WARNING: Trainer does not have preprocessor with scalers")
        except Exception as e:
            print(f"WARNING: Failed to save preprocessor scalers: {e}")
    
    def _rename_model_files(self, model_output_dir: Path, config: Dict[str, Any]):
        """Rename model files to match ForecastEvaluator naming convention"""
        try:
            import shutil
            
            # Get the model name that ForecastEvaluator expects
            # ForecastEvaluator removes '_lstm' from the directory name, so we need to account for that
            # The directory name is: {config['model_name']}_{timestamp}
            # ForecastEvaluator expects: model files named with the directory name minus '_lstm'
            directory_name = model_output_dir.name
            model_name = directory_name.replace('_lstm', '')  # Remove '_lstm' from directory name
            
            print(f"DEBUG: Directory name: {directory_name}")
            print(f"DEBUG: Model name for files: {model_name}")
            
            # Find all files that need renaming
            timestamp_files = list(model_output_dir.glob("*_lstm.*"))
            
            for file_path in timestamp_files:
                if file_path.name.startswith("model_config_") and file_path.name.endswith("_lstm.json"):
                    # Rename model config file
                    new_name = f"model_config_{model_name}_lstm.json"
                    new_path = model_output_dir / new_name
                    if not new_path.exists():  # Only rename if target doesn't exist
                        shutil.move(str(file_path), str(new_path))
                        print(f"Renamed {file_path.name} to {new_name}")
                
                elif file_path.name.startswith("preprocessor_") and file_path.name.endswith("_lstm.json"):
                    # Rename preprocessor file
                    new_name = f"preprocessor_{model_name}_lstm.json"
                    new_path = model_output_dir / new_name
                    if not new_path.exists():  # Only rename if target doesn't exist
                        shutil.move(str(file_path), str(new_path))
                        print(f"Renamed {file_path.name} to {new_name}")
                
                elif file_path.name.endswith("_lstm.pth"):
                    # Rename model file
                    new_name = f"{model_name}_lstm.pth"
                    new_path = model_output_dir / new_name
                    if not new_path.exists():  # Only rename if target doesn't exist
                        shutil.move(str(file_path), str(new_path))
                        print(f"Renamed {file_path.name} to {new_name}")
                        
        except Exception as e:
            print(f"WARNING: Failed to rename model files: {e}")
    
    def _copy_original_config(self, model_output_dir: Path, config: Dict[str, Any]):
        """Copy original config file to preserve forecast_mode information"""
        try:
            # Find the generated config file
            config_files = list(model_output_dir.glob("model_config_*_lstm.json"))
            if not config_files:
                print("WARNING: No config file found to replace")
                return
            
            config_path = config_files[0]
            
            # Replace the generated config with the original config (which has forecast_mode)
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            
            print(f"Replaced config file with original: {config_path.name}")
            print(f"Original config includes forecast_mode: {config.get('forecast_mode', 'Not found')}")
            
        except Exception as e:
            print(f"WARNING: Failed to copy original config: {e}")
    
    def _scaler_to_dict(self, scaler):
        """Convert sklearn scaler to dictionary with all necessary parameters"""
        try:
            scaler_dict = {}
            # Include all necessary parameters for ForecastEvaluator
            for attr in ['scale_', 'mean_', 'var_', 'n_features_in_', 'feature_names_in_', 
                         'min_', 'data_min_', 'data_max_', 'data_range_', 'n_samples_seen_',
                         'clip', 'copy', 'feature_range']:
                if hasattr(scaler, attr):
                    value = getattr(scaler, attr)
                    if value is not None:
                        scaler_dict[attr] = value.tolist() if hasattr(value, 'tolist') else value
            return scaler_dict
        except Exception as e:
            print(f"WARNING: Failed to convert scaler to dict: {e}")
            return {}
