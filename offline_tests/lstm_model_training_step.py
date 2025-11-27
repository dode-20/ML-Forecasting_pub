#!/usr/bin/env python3
"""
Offline LSTM Model Training Test

This test validates the LSTM model training functionality using the actual
LSTM service modules from the lstm_model directory. It tests the complete
training pipeline without requiring the Docker environment.

"""

import sys
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from typing import Optional
import os # Added for os.path.exists
import numpy as np # Added for np.isinf
load_dotenv(Path(__file__).parent / ".env")

# Add parent directories to path to import modules
sys.path.append(str(Path(__file__).parent.parent / "influxData_api" / "data"))

# Import the original influxDB_client
from influxDB_client import influxClient

# Import trainers using importlib to avoid module conflicts
import importlib.util

# Import One-Step trainer
onestep_path = Path(__file__).parent.parent / "lstm_model" / "service" / "src" / "train.py"
spec = importlib.util.spec_from_file_location("onestep_train", onestep_path)
onestep_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(onestep_module)
OneStepLSTMTrainer = onestep_module.LSTMTrainer

# Import Multi-Step trainer
multistep_path = Path(__file__).parent.parent / "lstm_model_multistep" / "service" / "src" / "train.py"
spec = importlib.util.spec_from_file_location("multistep_train", multistep_path)
multistep_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(multistep_module)
MultiStepLSTMTrainer = multistep_module.MultiStepLSTMTrainer

class OfflineLSTMModelTest:
    def __init__(self, config_file: str = "test_config.json", csv_file: Optional[str] = None, device: str = "cuda"):
        """
        Initialize the offline LSTM test with configuration and CSV file path.
        Args:
            config_file: Path to the JSON configuration file (same as UI)
            csv_file: Path to the training data CSV (optional, will auto-detect if None)
            device: Training device ("auto", "cpu", "cuda")
        """
        self.config_file = config_file
        self.config = self.load_config()
        
        # Determine forecast mode and select appropriate trainer
        self.forecast_mode = self.config.get("forecast_mode", {}).get("mode", "one-step")
        self.forecast_steps = self.config.get("forecast_mode", {}).get("forecast_steps", 1)
        
        print(f"[INFO] Forecast mode detected: {self.forecast_mode}")
        print(f"[INFO] Forecast steps: {self.forecast_steps}")
        
        # Select the appropriate trainer class
        if self.forecast_mode == "multi-step":
            print(f"[INFO] Using Multi-Step LSTM Model")
            self.LSTMTrainer = MultiStepLSTMTrainer
        else:
            print(f"[INFO] Using One-Step LSTM Model")
            self.LSTMTrainer = OneStepLSTMTrainer
        
        # Persist balancing flag from config for reuse
        try:
            self.use_balancing = bool(self.config.get('balancing', {}).get('use_balancing', False))
        except Exception:
            self.use_balancing = bool(self.config.get('use_balancing', False))
        self.client = influxClient(env_path=str(Path(__file__).parent / ".env"))
        self.device = device
        
        # Auto-detect training data if not provided
        if csv_file is None:
            self.csv_file = self._find_latest_training_data()
        else:
            # Allow passing a directory path or a file relative to results/training_data/{module_type}/cleanData
            csv_path = Path(csv_file)
            if not csv_path.is_absolute():
                # Get module_type from config
                module_type = self.config.get("module_type", "silicon")
                if module_type == "silicon":
                    base_dir = Path(__file__).parent.parent / "results" / "training_data" / "Silicon" / "cleanData"
                elif module_type == "perovskite":
                    base_dir = Path(__file__).parent.parent / "results" / "training_data" / "Perovskite" / "cleanData"
                else:
                    raise ValueError(f"Unsupported module_type: {module_type}. Must be 'silicon' or 'perovskite'")
                csv_path = base_dir / csv_path

            if csv_path.is_dir():
                # Select file inside directory based on data_resolution + weather flag
                self.csv_file = self._resolve_training_file_from_dir(csv_path)
            else:
                self.csv_file = csv_path
            
        if not self.csv_file.exists():
            raise FileNotFoundError(f"The specified CSV file does not exist: {self.csv_file}")
            
        # Output directory for this test run (new structure)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        self.output_dir = Path(__file__).parent.parent / "results" / "trained_models" / "lstm" / f"{self.timestamp}_lstm"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"LSTM model training results will be saved to: {self.output_dir}")
        print(f"Training device: {device}")
        print(f"Training data: {self.csv_file}")

    def _resolve_training_file_from_dir(self, dir_path: Path) -> Path:
        """
        Pick the correct training CSV inside a period folder based on
        config.data_resolution and weather_data.use_weatherData.
        """
        if not dir_path.exists():
            raise FileNotFoundError(f"Training data directory not found: {dir_path}")

        res = self.config.get("data_resolution", "5min")
        use_weather = self.config.get("weather_data", {}).get("use_weatherData", False)

        # Build preferred patterns (most specific first)
        patterns = []
        if use_weather:
            patterns.append(f"*_test_lstm_model_clean-{res}_weather-integrated.csv")
            # fallback without suffix
            patterns.append(f"*_test_lstm_model_clean-{res}.csv")
        else:
            patterns.append(f"*_test_lstm_model_clean-{res}.csv")
            # fallback with weather if present accidentally
            patterns.append(f"*_test_lstm_model_clean-{res}_weather-integrated.csv")

        # ultimate fallbacks
        patterns.append("*_test_lstm_model_clean.csv")
        patterns.append("*_test_lstm_model_clean*_weather-integrated.csv")

        for pat in patterns:
            matches = sorted(dir_path.glob(pat))
            if matches:
                print(f"[INFO] Selected training file by pattern '{pat}': {matches[-1]}")
                return matches[-1]

        raise FileNotFoundError(f"No training CSV found in {dir_path} for data_resolution={res}, use_weather={use_weather}")

    def _find_latest_training_data(self) -> Path:
        """Find the latest training data file from the pipeline based on module_type"""
        # Get module_type from config
        module_type = self.config.get("module_type", "silicon")
        
        # Build path based on module_type
        if module_type == "silicon":
            training_data_dir = Path(__file__).parent.parent / "results" / "training_data" / "Silicon" / "cleanData"
        elif module_type == "perovskite":
            training_data_dir = Path(__file__).parent.parent / "results" / "training_data" / "Perovskite" / "cleanData"
        else:
            raise ValueError(f"Unsupported module_type: {module_type}. Must be 'silicon' or 'perovskite'")
        
        if not training_data_dir.exists():
            raise FileNotFoundError(f"Training data directory not found: {training_data_dir}")
        
        # Look for the most recent training data file
        # Priority: 1. Files in subdirectories with pattern YYYYMMDD_YYYYMMDD_test_lstm_model_clean.csv
        #          2. Files directly in cleanData with pattern *_test_lstm_model_clean.csv
        
        latest_file = None
        latest_timestamp = None
        
        # Check subdirectories first (newer pipeline structure)
        for subdir in training_data_dir.iterdir():
            if subdir.is_dir():
                for file in subdir.glob("*_test_lstm_model_clean.csv"):
                    # Extract timestamp from filename or directory name
                    file_timestamp = self._extract_timestamp_from_path(file)
                    if file_timestamp and (latest_timestamp is None or file_timestamp > latest_timestamp):
                        latest_file = file
                        latest_timestamp = file_timestamp
        
        # If no file found in subdirectories, check root directory
        if latest_file is None:
            for file in training_data_dir.glob("*_test_lstm_model_clean.csv"):
                file_timestamp = self._extract_timestamp_from_path(file)
                if file_timestamp and (latest_timestamp is None or file_timestamp > latest_timestamp):
                    latest_file = file
                    latest_timestamp = file_timestamp
        
        if latest_file is None:
            raise FileNotFoundError(f"No training data files found in {training_data_dir}")
        
        print(f"Found latest training data for {module_type} modules: {latest_file}")
        return latest_file
    
    def _extract_timestamp_from_path(self, file_path: Path) -> Optional[datetime]:
        """Extract timestamp from file path for comparison"""
        try:
            # Try to extract from filename first (e.g., 20240901_20250625_test_lstm_model_clean.csv)
            filename = file_path.stem
            if '_test_lstm_model_clean' in filename:
                # Extract the date range part
                date_part = filename.split('_test_lstm_model_clean')[0]
                if '_' in date_part:
                    # Use the end date (second part after underscore)
                    end_date_str = date_part.split('_')[1]
                    return datetime.strptime(end_date_str, '%Y%m%d')
            
            # Fallback: try to extract from parent directory name
            parent_name = file_path.parent.name
            if '_' in parent_name:
                end_date_str = parent_name.split('_')[1]
                return datetime.strptime(end_date_str, '%Y%m%d')
                
        except (ValueError, IndexError):
            pass
        
        # If no valid timestamp found, use file modification time
        return datetime.fromtimestamp(file_path.stat().st_mtime)

    def load_config(self) -> dict:
        """
        Load the test configuration from a JSON file (same structure as UI).
        If the file does not exist, raise an error.
        """
        # Check if config_file is a full path or just filename
        if Path(self.config_file).is_absolute() or '/' in self.config_file or '\\' in self.config_file:
            # Full path provided
            config_path = Path(self.config_file)
        else:
            # Just filename provided, look in results/model_configs/
            config_path = Path(__file__).parent.parent / "results" / "model_configs" / self.config_file
            
        if not config_path.exists():
            raise FileNotFoundError(f"The specified config file does not exist: {config_path}")
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def test_data_loading(self) -> Optional[pd.DataFrame]:
        """Tests data loading from CSV file"""
        print("Loading training data...")
        try:
            # Selection of the correct file based on data_resolution + weather flag
            use_weather = self.config.get("weather_data", {}).get("use_weatherData", False)
            res = self.config.get("data_resolution", "5min")
            orig_path = Path(str(self.csv_file))
            dir_path = orig_path.parent

            try:
                selected = self._resolve_training_file_from_dir(dir_path)
                print(f"OK: Automatically selected training file ({res}, weather={use_weather}): {selected}")
                df = pd.read_csv(selected)
            except Exception as e:
                # Fallback: try to load the originally found file
                print(f"[WARN] Auto-selection failed: {e}. Trying to load original file: {orig_path}")
                df = pd.read_csv(orig_path)
            
            if df is None or df.empty:
                print(f"FAIL Data loading failed: Empty dataset")
                return None
            
            # DEBUG: Comprehensive data analysis
            print(f"[DEBUG] ===== TRAINING DATA ANALYSIS =====")
            print(f"[DEBUG] Data shape: {df.shape}")
            print(f"[DEBUG] Data columns: {list(df.columns)}")
            
            # Clean column names (remove trailing spaces)
            df.columns = df.columns.str.strip()
            print(f"[DEBUG] Cleaned columns: {list(df.columns)}")
            
            print(f"[DEBUG] Data types:")
            print(df.dtypes)
            print(f"[DEBUG] Data info:")
            print(df.info())
            print(f"[DEBUG] Data first 5 rows:")
            print(df.head())
            print(f"[DEBUG] Data last 5 rows:")
            print(df.tail())
            
            # Check for missing values
            print(f"[DEBUG] Missing values per column:")
            missing_data = df.isnull().sum()
            for col, missing_count in missing_data.items():
                if missing_count > 0:
                    print(f"[DEBUG]   {col}: {missing_count} missing values")
            
            # Check for infinite values
            print(f"[DEBUG] Infinite values per column:")
            for col in df.select_dtypes(include=[np.number]).columns:
                inf_count = np.isinf(df[col]).sum()
                if inf_count > 0:
                    print(f"[DEBUG]   {col}: {inf_count} infinite values")
            
            # Check output features specifically
            output_features = self.config.get("output", [])
            print(f"[DEBUG] Output features from config: {output_features}")
            for feature in output_features:
                if feature in df.columns:
                    feature_data = df[feature]
                    print(f"[DEBUG] Output feature '{feature}':")
                    print(f"[DEBUG]   min: {feature_data.min()}, max: {feature_data.max()}")
                    print(f"[DEBUG]   mean: {feature_data.mean()}, std: {feature_data.std()}")
                    print(f"[DEBUG]   unique values: {feature_data.nunique()}")
                    print(f"[DEBUG]   zero count: {len(feature_data[feature_data == 0])}")
                    print(f"[DEBUG]   non-zero count: {len(feature_data[feature_data != 0])}")
                else:
                    print(f"[WARN] Output feature '{feature}' not found in data!")
            
            print(f"OK Data loading: OK ({len(df)} rows, {len(df.columns)} columns)")
            print(f"  Columns: {list(df.columns)}")

            # Optional balancing (50/50) controlled via config
            if self.use_balancing:
                print(f"[DEBUG] Balancing enabled: creating 50/50 dataset for training")
                print(f"[DEBUG] Original data count: {len(df)} rows")
                print(f"[DEBUG] P statistics (raw data):")
                print(f"[DEBUG]   min: {df['P'].min()}, max: {df['P'].max()}")
                print(f"[DEBUG]   mean: {df['P'].mean()}, std: {df['P'].std()}")
                print(f"[DEBUG]   zero count: {len(df[df['P'] == 0])}")
                print(f"[DEBUG]   non-zero count: {len(df[df['P'] > 0])}")
                print(f"[DEBUG]   zero percentage: {len(df[df['P'] == 0])/len(df)*100:.1f}%")

                zero_data = df[df['P'] == 0].copy()
                power_data = df[df['P'] > 0].copy()
                target_zero_count = len(power_data)
                if len(zero_data) > target_zero_count:
                    zero_sampled = zero_data.sample(n=target_zero_count, random_state=42)
                else:
                    zero_sampled = zero_data
                df = pd.concat([power_data, zero_sampled]).sort_values('timestamp').reset_index(drop=True)

                print(f"[DEBUG] Balanced dataset:")
                print(f"[DEBUG]   - Power data: {len(power_data)} rows")
                print(f"[DEBUG]   - Zero data (sampled): {len(zero_sampled)} rows")
                print(f"[DEBUG]   - Total balanced: {len(df)} rows")
                print(f"[DEBUG]   - New zero percentage: {len(df[df['P'] == 0])/len(df)*100:.1f}%")
            else:
                print(f"[INFO] Balancing disabled. Using full dataset unchanged (recommended for baseline).")
                
            print(f"OK Data loading: OK ({len(df)} rows, {len(df.columns)} columns)")
            print(f"  Columns: {list(df.columns)}")
            return df
        except Exception as e:
            print(f"FAIL Data loading failed: {e}")
            return None

    def _load_weather_forecast_data(self, historical_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create weather forecast data for Weather-Informed mode.
        Uses the same dataset but shifts the weather data to simulate forecasts.
        """
        try:
            weather_features = self.config.get("weather_data", {}).get("weather_features", [])
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

    def test_model_training(self, df: pd.DataFrame) -> bool:
        """Tests the complete LSTM training pipeline"""
        print("Starting model training...")
        try:
            features = self.config["features"]
            time_features = self.config.get("time_features", [])
            
            # Add weather features if enabled
            weather_features = []
            if self.config.get("weather_data", {}).get("use_weatherData", True):
                weather_features = self.config["weather_data"].get("weather_features", [])
                print(f"  Weather features enabled: {weather_features}")
            else:
                print(f"  Weather features disabled")
            
            # Combine all features: time_features + features + weather_features
            all_features = time_features + features + weather_features
            
            # DEBUG: Feature analysis
            print(f"[DEBUG] ===== FEATURE ANALYSIS =====")
            print(f"[DEBUG] Time features: {time_features}")
            print(f"[DEBUG] PV features: {features}")
            print(f"[DEBUG] Weather features: {weather_features}")
            print(f"[DEBUG] All features: {all_features}")
            print(f"[DEBUG] Total features: {len(all_features)}")
            
            # Check if all features exist in data
            missing_features = [f for f in all_features if f not in df.columns]
            if missing_features:
                print(f"[WARN] Missing features in data: {missing_features}")
                print(f"[WARN] Available columns: {list(df.columns)}")
            
            training_settings = {
                "model_name": self.config["model_name"],
                "features": all_features,
                "output": self.config["output"],
                "batch_size": self.config["batch_size"],
                "epochs": self.config["epochs"],
                "learning_rate": self.config["learning_rate"],
                "loss_function": self.config["loss_function"],
                "shuffle": self.config["shuffle"],
                "use_validation_set": self.config["validation_set"]["use_validation_set"],
                "validation_split": self.config["validation_set"]["validation_split"],
                "sequence_length": self.config.get("sequence_length", 864),
                # NEW: pass through data resolution (defaults to 5min if missing)
                "data_resolution": self.config.get("data_resolution", "5min"),
                "use_balancing": self.use_balancing,
                # NEW: pass through forecast mode settings
                "forecast_mode": self.config.get("forecast_mode", {"mode": "one-step", "forecast_steps": 1}),
                # NEW: pass through model architecture parameters
                "hidden_size": self.config.get("hidden_size", 64),
                "num_layers": self.config.get("num_layers", 2),
                "dropout": self.config.get("dropout", 0.1),
                # NEW: pass through early stopping patience
                "early_stopping_patience": self.config.get("early_stopping_patience", 10)
            }
            self.training_settings = training_settings
            
            # DEBUG: Training settings analysis
            print(f"[DEBUG] ===== TRAINING SETTINGS ANALYSIS =====")
            print(f"[DEBUG] Training settings: {training_settings}")
            print(f"[DEBUG] Output features: {training_settings['output']}")
            print(f"[DEBUG] Input features: {training_settings['features']}")
            print(f"[DEBUG] Expected input size: {len(training_settings['features'])}")
            print(f"[DEBUG] Expected output size: {len(training_settings['output'])}")
            
            print(f"  Training settings: {training_settings}")
            print(f"  Feature breakdown:")
            print(f"    Time features: {time_features}")
            print(f"    PV features: {features}")
            print(f"    Weather features: {weather_features}")
            print(f"    Total features: {len(all_features)}")
            
            # Initialize the appropriate LSTMTrainer with external timestamp and directory
            trainer = self.LSTMTrainer(
                training_settings, 
                device=self.device,
                external_timestamp=self.timestamp,
                external_model_dir=self.output_dir
            )
            print(f"OK LSTMTrainer initialized on device: {trainer.device}")
            print(f"  Model type: {self.forecast_mode}")
            print(f"  Forecast steps: {self.forecast_steps}")
            
            # DEBUG: Pre-training analysis
            print(f"[DEBUG] ===== PRE-TRAINING ANALYSIS =====")
            print(f"[DEBUG] Data shape before training: {df.shape}")
            print(f"[DEBUG] Data columns before training: {list(df.columns)}")
            print(f"[DEBUG] Forecast mode: {self.forecast_mode}")
            print(f"[DEBUG] Forecast steps: {self.forecast_steps}")
            
            # For Weather-Informed mode, use the same dataset for both historical and "forecast" data
            weather_forecast_data = None
            if self.config.get("weather_data", {}).get("use_weatherData", False):
                print("  Using same dataset for Weather-Informed mode (training simulation)")
                weather_forecast_data = df  # Use same dataset as "forecast" data
                print(f"  OK Using historical data as weather forecast simulation: {weather_forecast_data.shape}")
            
            # Run training
            print("  Starting training process...")
            training_history = trainer.train(df, weather_forecast_data)
            
            # DEBUG: Post-training analysis
            print(f"[DEBUG] ===== POST-TRAINING ANALYSIS =====")
            if training_history:
                print(f"[DEBUG] Training history keys: {list(training_history.keys())}")
                if "train_loss" in training_history:
                    train_losses = training_history["train_loss"]
                    print(f"[DEBUG] Training losses: {len(train_losses)} epochs")
                    print(f"[DEBUG] Training loss progression: {train_losses[:5]}...{train_losses[-5:]}")
                    print(f"[DEBUG] Final training loss: {train_losses[-1]:.8f}")
                if "val_loss" in training_history and training_history["val_loss"]:
                    val_losses = training_history["val_loss"]
                    print(f"[DEBUG] Validation losses: {len(val_losses)} epochs")
                    print(f"[DEBUG] Validation loss progression: {val_losses[:5]}...{val_losses[-5:]}")
                    print(f"[DEBUG] Final validation loss: {val_losses[-1]:.8f}")
            
            if training_history and "train_loss" in training_history:
                print(f"OK Model training: OK")
                print(f"  Final training loss: {training_history['train_loss'][-1]:.6f}")
                if "val_loss" in training_history and training_history["val_loss"]:
                    print(f"  Final validation loss: {training_history['val_loss'][-1]:.6f}")
                self.save_training_results(training_history, trainer, self.training_settings)
                return True
            else:
                print("FAIL Model training failed: No training history returned")
                return False
        except Exception as e:
            print(f"FAIL Model training failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def save_training_results(self, training_history: dict, trainer, training_settings: dict):
        """Save training results and model information to the output directory (new structure)"""
        # Use the same timestamp for all files
        timestamp = self.timestamp
        # Save model weights if possible
        model_weights_path = self.output_dir / f"{timestamp}_lstm.pth"
        if hasattr(trainer, 'model') and trainer.model is not None:
            try:
                import torch
                torch.save(trainer.model.state_dict(), model_weights_path)
                print(f"Model weights saved: {model_weights_path}")
            except Exception as e:
                print(f"Warning: Could not save model weights: {e}")
        # Save training history
        history_path = self.output_dir / f"training_history_{timestamp}_lstm.json"
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=4)
        # Save model configuration with safe attribute access
        config_path = self.output_dir / f"model_config_{timestamp}_lstm.json"
        # Safely get model attributes
        model_attrs = {}
        if trainer.model is not None:
            try:
                model_attrs = {
                    "input_size": getattr(trainer.model, 'input_size', None),
                    "hidden_size": getattr(trainer.model, 'hidden_size', None),
                    "num_layers": getattr(trainer.model, 'num_layers', None),
                    "output_size": getattr(trainer.model, 'output_size', None),
                    "dropout": getattr(trainer.model, 'dropout', None)
                }
            except Exception as e:
                print(f"Warning: Could not access model attributes: {e}")
                model_attrs = {
                    "input_size": None,
                    "hidden_size": None,
                    "num_layers": None,
                    "output_size": None,
                    "dropout": None
                }
        else:
            print("Warning: Trainer model is None")
            model_attrs = {
                "input_size": None,
                "hidden_size": None,
                "num_layers": None,
                "output_size": None,
                "dropout": None
            }
        # Get combined features (time_features + features + weather_features)
        time_features = self.config.get("time_features", [])
        features = self.config["features"]
        
        # Add weather features if enabled
        weather_features = []
        if self.config.get("weather_data", {}).get("use_weatherData", True):
            weather_features = self.config["weather_data"].get("weather_features", [])
        
        all_features = time_features + features + weather_features
        
        model_config = {
            "model_name": self.config["model_name"],
            "device": str(trainer.device),
            **model_attrs,
            "sequence_length": training_settings["sequence_length"],
            "forecast_mode": self.config.get("forecast_mode", {"mode": "one-step", "forecast_steps": 1}),
            "training_settings": {
                "features": all_features,  # Use combined features
                "output": self.config["output"],
                "batch_size": self.config["batch_size"],
                "epochs": self.config["epochs"],
                "learning_rate": self.config["learning_rate"],
                "loss_function": self.config["loss_function"],
                "shuffle": self.config["shuffle"],
                "use_validation_set": self.config["validation_set"]["use_validation_set"],
                "validation_split": self.config["validation_set"]["validation_split"],
                "sequence_length": training_settings["sequence_length"],
                "data_resolution": training_settings["data_resolution"],
                "forecast_mode": self.config.get("forecast_mode", {"mode": "one-step", "forecast_steps": 1})
            },
            "weather_data": {
                "use_weatherData": self.config.get("weather_data", {}).get("use_weatherData", False),
                "weather_features": weather_features
            }
        }
        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=4)
        # Create training summary
        summary_path = self.output_dir / f"training_summary_{timestamp}_lstm.txt"
        with open(summary_path, 'w') as f:
            f.write("=== LSTM Training Test Summary ===\n")
            f.write(f"Test time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model Name: {self.config['model_name']}\n")
            f.write(f"Model Type: {self.forecast_mode}\n")
            f.write(f"Forecast Steps: {self.forecast_steps}\n")
            f.write(f"Sequence Length: {self.config.get('sequence_length', 8)}\n")
            f.write(f"Sequence-to-Forecast Ratio: {self.config.get('sequence_length', 8) / self.forecast_steps:.2f}\n")
            f.write(f"Device: {trainer.device}\n")
            f.write(f"PV Features: {self.config['features']}\n")
            f.write(f"Time Features: {self.config.get('time_features', [])}\n")
            
            # Add weather features info
            weather_enabled = self.config.get("weather_data", {}).get("use_weatherData", False)
            weather_features = self.config.get("weather_data", {}).get("weather_features", [])
            f.write(f"Weather Data Enabled: {weather_enabled}\n")
            if weather_enabled:
                f.write(f"Weather Features: {weather_features}\n")
            
            f.write(f"Total Features: {len(all_features)}\n")
            f.write(f"Output: {self.config['output']}\n")
            f.write(f"Epochs: {self.config['epochs']}\n")
            f.write(f"Batch Size: {self.config['batch_size']}\n")
            f.write(f"Learning Rate: {self.config['learning_rate']}\n")
            f.write(f"Loss Function: {self.config['loss_function']}\n")
            f.write(f"Final Training Loss: {training_history['train_loss'][-1]:.6f}\n")
            if "val_loss" in training_history and training_history["val_loss"]:
                f.write(f"Final Validation Loss: {training_history['val_loss'][-1]:.6f}\n")
            f.write(f"Training History saved to: {history_path}\n")
            f.write(f"Model Config saved to: {config_path}\n")
            f.write(f"Model Weights saved to: {model_weights_path}\n")
        print(f"Training results saved to: {self.output_dir}")
        print(f"  - Model weights: {model_weights_path}")
        print(f"  - Training history: {history_path}")
        print(f"  - Model config: {config_path}")
        print(f"  - Summary: {summary_path}")

    def run_full_test(self) -> bool:
        """Runs the complete LSTM training test"""
        print("=" * 50)
        print("LSTM MODEL TRAINING TEST")
        print("=" * 50)
        print(f"Configuration: {self.config_file}")
        print(f"Device: {self.device}")
        print(f"Training data: {self.csv_file}")
        
        # Test 1: Data loading
        df = self.test_data_loading()
        if df is None:
            return False
        
        # Test 2: Model training
        success = self.test_model_training(df)
        if not success:
            return False
        
        print("\n" + "=" * 50)
        print("OK ALL TESTS PASSED")
        print("=" * 50)
        return True

def main():
    """Main function to run the LSTM training test with command line arguments"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run LSTM model training test")
    parser.add_argument("config_file", nargs="?", default="test_lstm_model_settings.json",
                       help="Path to model configuration file")
    parser.add_argument("--csv-file", default=None,
                       help="Path to training data CSV (optional, will auto-detect if not provided)")
    parser.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"],
                       help="Training device")
    
    args = parser.parse_args()
    
    try:
        test = OfflineLSTMModelTest(
            config_file=args.config_file,
            csv_file=args.csv_file,
            device=args.device
        )
        
        success = test.run_full_test()
        if success:
            print("SUCCESS: LSTM model training test completed successfully")
            sys.exit(0)
        else:
            print("ERROR: LSTM model training test failed")
            sys.exit(1)
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # Direct configuration for easy testing
    # You can modify these paths as needed
    
    # Configuration file (relative to results/model_configs/)
    CONFIG_FILE = "test_lstm_model_settings.json"
    
    # Load config to determine correct data resolution and paths
    try:
        config_path = Path(__file__).parent.parent / "results" / "model_configs" / CONFIG_FILE
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Extract parameters from config
        module_type = config.get('module_type', 'silicon').capitalize()
        data_resolution = config.get('data_resolution', '5min')
        start_date = config.get('date_selection', {}).get('start', '2024-08-17').replace('-', '')
        end_date = config.get('date_selection', {}).get('end', '2025-06-22').replace('-', '')
        model_name = config.get('model_name', 'test_lstm_model')
        use_weather = config.get('weather_data', {}).get('use_weatherData', False)
        
        # Build dynamic CSV file path based on config
        date_range = f"{start_date}_{end_date}"
        weather_suffix = "_weather-integrated" if use_weather else ""
        CSV_FILE = f"{date_range}/{date_range}_{model_name}_clean-{data_resolution}{weather_suffix}.csv"
        
        print(f"[INFO] Config-based path selection:")
        print(f"[INFO]   Module type: {module_type}")
        print(f"[INFO]   Data resolution: {data_resolution}")
        print(f"[INFO]   Use weather: {use_weather}")
        print(f"[INFO]   CSV file: {CSV_FILE}")
        
    except Exception as e:
        print(f"[WARN] Failed to load config for dynamic path selection: {e}")
        print("[WARN] Using fallback hardcoded path...")
        # Fallback to hardcoded path
        CSV_FILE = "20240901_20250725/20240901_20250725_test_lstm_model_clean-1h_weather-integrated.csv"
    
    # Training device
    DEVICE = "cuda"  # Options: "auto", "cpu", "cuda"
    
    print("=" * 60)
    print("LSTM MODEL TRAINING - DIRECT CONFIGURATION")
    print("=" * 60)
    print(f"Config file: {CONFIG_FILE}")
    print(f"CSV file: {CSV_FILE or 'auto-detect'}")
    print(f"Device: {DEVICE}")
    print("=" * 60)
    
    try:
        test = OfflineLSTMModelTest(
            config_file=CONFIG_FILE,
            csv_file=CSV_FILE,
            device=DEVICE
        )
        
        success = test.run_full_test()
        if success:
            print("SUCCESS: LSTM model training test completed successfully")
        else:
            print("ERROR: LSTM model training test failed")
            
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}")
        import traceback
        traceback.print_exc()