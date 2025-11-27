#!/usr/bin/env python3
"""
Forecast Evaluator - Comprehensive Model Evaluation

This script performs a complete analysis of LSTM models:
- Loads trained models and test data
- Automatically detects model type (one-step vs multi-step)
- Uses appropriate evaluation method for each model type
- Calculates comprehensive metrics (RMSE, MAE, MAPE, R², Skill Score)
- Creates visualizations (Scatter plots, time series comparison, error distribution)
- Generates detailed reports with method information
- Walk-Forward validation on historical data
- Tests model performance over different time periods
- Analyzes seasonal patterns and trends

Usage:
    python forecast_evaluator.py --model 20250707_1152
    python forecast_evaluator.py --all-models
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages

# Import Analytical Packages
try:
    from analytical_packages import (
        VariabilityAnalysis,
        analyze_cv_prediction_quality,
        RampRateAnalysis,
        analyze_ramp_rate_prediction_quality,
        IntermittencyAnalysis,
        analyze_intermittency_prediction_quality,
        PowerLevelAnalysis,
        analyze_power_level_prediction_quality
    )
    ANALYTICAL_PACKAGES_AVAILABLE = True
except ImportError:
    print("[WARN] Analytical packages not available")
    ANALYTICAL_PACKAGES_AVAILABLE = False
import plotly.express as px
import argparse

# Add project paths - will be determined dynamically based on model type
import torch

# Import evaluator components
try:
    # Try relative imports first (when used as module)
    from .base.evaluator_factory import EvaluatorFactory
    from .base.base_evaluator import BaseEvaluator
except ImportError:
    # Fall back to absolute imports (when run directly)
    from base.evaluator_factory import EvaluatorFactory
    from base.base_evaluator import BaseEvaluator


class ForecastEvaluator:
    """
    Comprehensive forecast evaluation for LSTM models
    """
    
    def __init__(self, 
                 model_path: str,
                 silicon_data_path: str = None,
                 perovskite_data_path: str = None,
                 output_dir: str = "results/forecast_evaluation",
                 exclude_periods: List[Tuple[str, str]] = None):
        """
        Initialize the Forecast Evaluator
        
        Args:
            model_path: Path to specific model directory (e.g. "results/trained_models/lstm/20250707_1152_lstm")
            silicon_data_path: Path to Silicon validation dataset (optional, will use weather-integrated data if available)
            perovskite_data_path: Path to Perovskite validation dataset (optional, will use weather-integrated data if available)
            output_dir: Output directory for results
            exclude_periods: List of tuples with (start_date, end_date) strings to exclude from validation.
                           Format: [("2024-01-01", "2024-01-05"), ("2024-02-10", "2024-02-15")]
        """
        self.model_path = Path(model_path)
        self.silicon_data_path = Path(silicon_data_path) if silicon_data_path else None
        self.perovskite_data_path = Path(perovskite_data_path) if perovskite_data_path else None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize exclude periods
        self.exclude_periods = exclude_periods or []
        if self.exclude_periods:
            print(f"[INFO] Excluding {len(self.exclude_periods)} time periods from validation:")
            for i, (start, end) in enumerate(self.exclude_periods):
                print(f"  {i+1}. {start} to {end}")
        
        # Model components
        self.model = None
        self.preprocessor = None
        self.config = None
        self.model_name = None
        self.evaluator = None  # Will be set based on model type
        
        # Evaluation results
        self.evaluation_results = {}
        self.timestamps = None
        
        # Analytical packages results
        self.analytical_results = {}
        
    def load_model(self):
        """Load the specified model"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path not found: {self.model_path}")
        
        # Extract model name from path
        # The path format is: baseline_lstm_model_batch_size_32_20250927_2133
        # We need to remove the timestamp part, not the _lstm part
        path_parts = self.model_path.name.split('_')
        if len(path_parts) >= 2 and path_parts[-1].isdigit() and path_parts[-2].isdigit():
            # Remove the last two parts (timestamp)
            self.model_name = '_'.join(path_parts[:-2])
        else:
            # Fallback: remove _lstm as before
            self.model_name = self.model_path.name.replace('_lstm', '')
        
        print(f"[INFO] Loading model from: {self.model_path}")
        print(f"[DEBUG] Extracted model_name: {self.model_name}")
        
        # Load configuration first
        config_path = self.model_path / f"model_config_{self.model_name}_lstm.json"
        print(f"[DEBUG] Looking for config file: {config_path}")
        print(f"[DEBUG] Config file exists: {config_path.exists()}")
        if not config_path.exists():
            # List all files in the directory for debugging
            print(f"[DEBUG] Files in model directory:")
            for file_path in self.model_path.rglob('*'):
                if file_path.is_file():
                    print(f"  - {file_path}")
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, "r") as f:
            self.config = json.load(f)
        
        # Determine model type and set correct path
        forecast_mode = self.config.get("forecast_mode", {"mode": "one-step", "forecast_steps": 1})
        model_mode = forecast_mode.get("mode", "one-step")
        
        if model_mode == "multi-step":
            src_path = Path(__file__).parent.parent.parent / "lstm_model_multistep" / "service" / "src"
            print(f"[INFO] Using Multi-Step model path: {src_path}")
        else:
            src_path = Path(__file__).parent.parent.parent / "lstm_model" / "service" / "src"
            print(f"[INFO] Using One-Step model path: {src_path}")
        
        # Add the correct model path to sys.path
        sys.path.insert(0, str(src_path))
        
        # Define features strictly from the saved training settings
        training_settings = self.config.get('training_settings', {})
        TRAINING_FEATURES = training_settings.get("features", [])
        
        # Fallback: if no features in training_settings, try direct config
        if not TRAINING_FEATURES:
            TRAINING_FEATURES = self.config.get("features", [])
            # Also add time features and weather features if available
            time_features = self.config.get("time_features", [])
            weather_features = []
            if self.config.get("weather_data", {}).get("use_weatherData", False):
                weather_features = self.config.get("weather_data", {}).get("weather_features", [])
            TRAINING_FEATURES = TRAINING_FEATURES + time_features + weather_features
        ALL_FEATURES = list(dict.fromkeys(TRAINING_FEATURES))

        # Informative logging (no auto-augmentation)
        print(f"  Feature breakdown:")
        print(f"    Training features (used exactly as saved): {TRAINING_FEATURES}")
        print(f"    Total features: {len(ALL_FEATURES)}")
        
        sequence_length = training_settings.get("sequence_length", self.config.get("sequence_length", 48))
        
        # Import DataPreprocessor using importlib.util to ensure correct path
        import importlib.util
        
        if "forecast_mode" in self.config and self.config["forecast_mode"].get("mode") == "multi-step":
            # Use DataPreprocessor from lstm_model_multistep
            preprocess_path = Path(__file__).parent.parent.parent / "lstm_model_multistep" / "service" / "src" / "preprocess.py"
            spec = importlib.util.spec_from_file_location("multistep_preprocess", preprocess_path)
            preprocess_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(preprocess_module)
            DataPreprocessor = preprocess_module.DataPreprocessor
            print(f"[INFO] Using DataPreprocessor from lstm_model_multistep for multi-step model")
        else:
            # Use DataPreprocessor from lstm_model (one-step)
            preprocess_path = Path(__file__).parent.parent.parent / "lstm_model" / "service" / "src" / "preprocess.py"
            spec = importlib.util.spec_from_file_location("onestep_preprocess", preprocess_path)
            preprocess_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(preprocess_module)
            DataPreprocessor = preprocess_module.DataPreprocessor
            print(f"[INFO] Using DataPreprocessor from lstm_model for one-step model")
        
        # Load preprocessor
        try:
            self.preprocessor = DataPreprocessor(
                features=ALL_FEATURES,
                output_features=training_settings.get("output", self.config.get("output", ["P_normalized"])),
                sequence_length=sequence_length,
                settings=self.config  # Pass the full config as settings
            )
            
            print(f"[DEBUG] Preprocessor initialized successfully")
            print(f"[DEBUG] Preprocessor type: {type(self.preprocessor)}")
            print(f"[DEBUG] Preprocessor has feature_scalers: {hasattr(self.preprocessor, 'feature_scalers')}")
            if hasattr(self.preprocessor, 'feature_scalers'):
                print(f"[DEBUG] Preprocessor feature_scalers: {self.preprocessor.feature_scalers}")
            else:
                print(f"[ERROR] Preprocessor does not have feature_scalers attribute!")
        except Exception as e:
            print(f"[ERROR] Failed to initialize preprocessor: {e}")
            raise

        # Resolution from config (documentation only). No resampling here; we will
        # select the correct weather-integrated CSV based on this for evaluation inputs.
        self.data_resolution = self.config.get("data_resolution", self.config.get("training_settings", {}).get("data_resolution", "5min"))
        
        # Load preprocessor parameters
        preprocessor_path = self.model_path / f"preprocessor_{self.model_name}_lstm.json"
        print(f"[DEBUG] Looking for preprocessor file: {preprocessor_path}")
        print(f"[DEBUG] Preprocessor file exists: {preprocessor_path.exists()}")
        
        if preprocessor_path.exists():
            with open(preprocessor_path, "r") as f:
                scaler_params = json.load(f)
            
            print(f"[DEBUG] Loaded scaler_params keys: {list(scaler_params.keys())}")
            print(f"[DEBUG] Feature scalers keys: {list(scaler_params.get('feature_scalers', {}).keys())}")
            print(f"[DEBUG] Preprocessor feature_scalers keys: {list(self.preprocessor.feature_scalers.keys())}")
            
            # Set preprocessor parameters
            for f_name, params in scaler_params.get("feature_scalers", {}).items():
                if f_name in self.preprocessor.feature_scalers:
                    scaler = self.preprocessor.feature_scalers[f_name]
                    print(f"[DEBUG] Loading scaler for feature: {f_name}")
                    for k, v in params.items():
                        if k in ["min_", "scale_", "data_min_", "data_max_", "data_range_", "n_samples_seen_"]:
                            if k == "n_samples_seen_":
                                setattr(scaler, k, int(v))
                            else:
                                setattr(scaler, k, np.array(v, dtype=np.float64))
                else:
                    print(f"[WARNING] Feature {f_name} not found in preprocessor.feature_scalers")
            
            for f_name, params in scaler_params.get("output_scalers", {}).items():
                if f_name in self.preprocessor.output_scalers:
                    scaler = self.preprocessor.output_scalers[f_name]
                    print(f"[DEBUG] Loading output scaler for feature: {f_name}")
                    for k, v in params.items():
                        if k in ["min_", "scale_", "data_min_", "data_max_", "data_range_", "n_samples_seen_"]:
                            if k == "n_samples_seen_":
                                setattr(scaler, k, int(v))
                            else:
                                setattr(scaler, k, np.array(v, dtype=np.float64))
                else:
                    print(f"[WARNING] Output feature {f_name} not found in preprocessor.output_scalers")
            
            print(f"[DEBUG] After loading scalers:")
            print(f"[DEBUG] Preprocessor feature_scalers: {list(self.preprocessor.feature_scalers.keys())}")
            print(f"[DEBUG] Preprocessor output_scalers: {list(self.preprocessor.output_scalers.keys())}")
        else:
            print(f"[ERROR] Preprocessor file not found: {preprocessor_path}")
            raise FileNotFoundError(f"Preprocessor file not found: {preprocessor_path}")

            # DEBUG: Ausgabe der P-Scaler-Parameter
            try:
                if "P" in self.preprocessor.output_scalers:
                    p_scaler = self.preprocessor.output_scalers["P"]
                    print("[DEBUG] Loaded output scaler for 'P':")
                    for attr in ["min_", "scale_", "data_min_", "data_max_", "data_range_", "n_samples_seen_"]:
                        val = getattr(p_scaler, attr, None)
                        print(f"[DEBUG]   P.{attr}: {val}")
                else:
                    print("[WARN] No output scaler found for 'P' in preprocessor")
            except Exception as e:
                print(f"[WARN] Could not print 'P' scaler parameters: {e}")
        
        # Load model
        model_file = self.model_path / f"{self.model_name}_lstm.pth"
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
            
        # Determine model type and import correct LSTMModel
        forecast_mode = self.config.get("forecast_mode", {"mode": "one-step", "forecast_steps": 1})
        model_mode = forecast_mode.get("mode", "one-step")
        
        # Import LSTMModel dynamically based on model type
        import importlib.util
        
        if model_mode == "multi-step":
            model_path = Path(__file__).parent.parent.parent / "lstm_model_multistep" / "service" / "src" / "model.py"
            print(f"[INFO] Using Multi-Step model: {model_path}")
        else:
            model_path = Path(__file__).parent.parent.parent / "lstm_model" / "service" / "src" / "model.py"
            print(f"[INFO] Using One-Step model: {model_path}")
        
        spec = importlib.util.spec_from_file_location("model", model_path)
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)
        LSTMModel = model_module.LSTMModel
        
        # Initialize model with appropriate parameters based on model type
        if model_mode == "multi-step":
            forecast_steps = forecast_mode.get("forecast_steps", 1)
            print(f"[INFO] Multi-step model detected: {forecast_steps} forecast steps")
            self.model = LSTMModel(
                input_size=len(ALL_FEATURES),
                hidden_size=self.config["hidden_size"],
                num_layers=self.config["num_layers"],
                output_size=len(training_settings.get("output", ["P"])),
                dropout=self.config["dropout"],
                forecast_steps=forecast_steps
            )
        else:
            print(f"[INFO] One-step model detected")
            self.model = LSTMModel(
                input_size=len(ALL_FEATURES),
                hidden_size=self.config["hidden_size"],
                num_layers=self.config["num_layers"],
                output_size=len(training_settings.get("output", ["P"])),
                dropout=self.config["dropout"]
            )
        self.model.load_state_dict(torch.load(model_file, map_location=torch.device("cpu")))
        self.model.eval()
        
        # Create appropriate evaluator based on model type
        self.evaluator = EvaluatorFactory.create_evaluator(
            self.model, self.preprocessor, self.config, self.model_name
        )
        
        print(f"[OK] Model {self.model_name} loaded successfully")
        print(f"[OK] Using evaluator: {self.evaluator.__class__.__name__}")
        return ALL_FEATURES
    
    def load_weather_integrated_data(self, features: List[str], module_type: str = None) -> pd.DataFrame:
        """Load weather-integrated data for evaluation"""
        # Try to find validation data first
        if module_type == "silicon" and self.silicon_data_path and self.silicon_data_path.exists():
            data_file = self.silicon_data_path
            print(f"[INFO] Using provided Silicon validation dataset: {data_file}")
        elif module_type == "perovskite" and self.perovskite_data_path and self.perovskite_data_path.exists():
            data_file = self.perovskite_data_path
            print(f"[INFO] Using provided Perovskite validation dataset: {data_file}")
        else:
            # Auto-detect weather-integrated file based on resolution and module_type
            # Get module_type from parameter or config
            if module_type is None:
                module_type = self.config.get("training_settings", {}).get("module_type", "silicon")
            
            # Build path based on module_type
            if module_type == "silicon":
                base_dir = Path("results/training_data/Silicon/cleanData")
            elif module_type == "perovskite":
                base_dir = Path("results/training_data/Perovskite/cleanData")
            else:
                raise ValueError(f"Unsupported module_type: {module_type}. Must be 'silicon' or 'perovskite'")
            
            res = getattr(self, 'data_resolution', '5min')
            search_patterns = [
                f"*_test_lstm_model_clean-{res}_weather-integrated.csv",
                f"*_test_lstm_model_clean-{res}.csv",
                f"*_test_lstm_model_clean_weather-integrated.csv",
                f"*_test_lstm_model_clean.csv",
            ]
            candidates = []
            if base_dir.exists():
                for sub in sorted([p for p in base_dir.iterdir() if p.is_dir()], reverse=True):
                    for pat in search_patterns:
                        candidates.extend(list(sub.glob(pat)))
            if not candidates:
                raise FileNotFoundError(f"No weather-integrated data found in {base_dir} for resolution {res} and module_type {module_type}")
            data_file = candidates[0]
            print(f"[INFO] Auto-detected {module_type.upper()} data: {data_file}")
        
        print(f"[INFO] Loading {module_type.upper()} weather-integrated data from: {data_file}")
        df = pd.read_csv(data_file)
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        elif '_time' in df.columns:
            df['timestamp'] = pd.to_datetime(df['_time'])
            df = df.rename(columns={'_time': 'timestamp'})
        
        # DEBUG: Grundlegende P-Statistiken der Rohdatei
        if 'P' in df.columns:
            try:
                p_series = pd.to_numeric(df['P'], errors='coerce')
                non_zero_count = int((p_series > 0).sum())
                print(f"[DEBUG] Raw CSV 'P' stats → min: {p_series.min():.4f}, max: {p_series.max():.4f}, mean: {p_series.mean():.4f}, >0 count: {non_zero_count}")
                nz = df[p_series > 0]
                if not nz.empty:
                    print(f"[DEBUG] First non-zero 'P' rows:\n{nz[['timestamp','P']].head(5)}")
                else:
                    print("[DEBUG] No non-zero 'P' found in loaded CSV")
            except Exception as e:
                print(f"[WARN] Could not compute raw 'P' stats: {e}")
        
        # Validate required features
        available_features = [f for f in features if f in df.columns]
        missing_features = [f for f in features if f not in df.columns]
        
        # DEBUG: Show feature validation details
        print(f"[DEBUG] {module_type.upper()} Feature validation:")
        print(f"  - Required features: {features}")
        print(f"  - Available features: {available_features}")
        print(f"  - Missing features: {missing_features}")
        print(f"  - All CSV columns: {list(df.columns)}")
        
        if missing_features:
            print(f"[WARN] Missing features: {missing_features}")
            print(f"[INFO] Available features: {list(df.columns)}")
        
        print(f"[INFO] Loaded {len(df)} records with {len(available_features)}/{len(features)} features")
        
        # Use the entire provided dataset for evaluation (no time filtering)
        if len(df) > 0:
            df = df.sort_values('timestamp')
            print(f"[INFO] Using entire validation dataset: {len(df)} records from {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        # Apply exclusion periods filter
        df = self._filter_excluded_periods(df, module_type)
        
        return df
    
    def load_both_module_data(self, features=None):
        """Load weather-integrated data for both Silicon and Perovskite modules"""
        print("[INFO] Loading data for both Silicon and Perovskite modules...")
        
        # Load Silicon data
        try:
            si_data = self.load_weather_integrated_data(features=features, module_type="silicon")
            print(f"[INFO] Successfully loaded Silicon data: {len(si_data)} rows")
        except Exception as e:
            print(f"[WARN] Could not load Silicon data: {e}")
            si_data = None
        
        # Load Perovskite data
        try:
            pvk_data = self.load_weather_integrated_data(features=features, module_type="perovskite")
            print(f"[INFO] Successfully loaded Perovskite data: {len(pvk_data)} rows")
        except Exception as e:
            print(f"[WARN] Could not load Perovskite data: {e}")
            pvk_data = None
        
        return si_data, pvk_data

    def _find_common_start_time(self, si_data: pd.DataFrame, pvk_data: pd.DataFrame) -> pd.Timestamp:
        """
        Find the common start time where both Silicon and Perovskite datasets have data
        
        Args:
            si_data: Silicon dataset
            pvk_data: Perovskite dataset
            
        Returns:
            Common start timestamp
        """
        if si_data is None or pvk_data is None:
            raise ValueError("Both Silicon and Perovskite datasets must be provided")
        
        # Get start times for both datasets
        si_start = si_data['timestamp'].min()
        pvk_start = pvk_data['timestamp'].min()
        
        print(f"[INFO] Silicon dataset starts at: {si_start}")
        print(f"[INFO] Perovskite dataset starts at: {pvk_start}")
        
        # Find the later start time (where both datasets have data)
        common_start = max(si_start, pvk_start)
        
        print(f"[INFO] Common start time for both datasets: {common_start}")
        
        return common_start
    
    def _synchronize_timestamps(self, evaluation_results: Dict[str, Any], common_start: pd.Timestamp) -> Dict[str, Any]:
        """
        Synchronize timestamps in evaluation results to start from common_start
        
        Args:
            evaluation_results: Results from evaluation
            common_start: Common start timestamp
            
        Returns:
            Synchronized evaluation results
        """
        print(f"[INFO] Synchronizing timestamps to start from {common_start}")
        
        synchronized_results = {}
        
        for module_type, module_data in evaluation_results.items():
            if module_data is None or 'timestamps' not in module_data:
                synchronized_results[module_type] = module_data
                continue
            
            timestamps = module_data['timestamps']
            predictions = module_data.get('predictions')
            actuals = module_data.get('actuals')
            
            # Convert timestamps to pandas Series if it's a list
            if isinstance(timestamps, list):
                timestamps = pd.Series(timestamps)
            
            # Find the index where timestamps >= common_start
            mask = timestamps >= common_start
            start_idx = mask.idxmax() if mask.any() else 0
            
            print(f"[INFO] {module_type}: Starting from index {start_idx} (timestamp: {timestamps.iloc[start_idx]})")
            
            # Filter data to start from common_start
            synchronized_timestamps = timestamps[mask]
            synchronized_predictions = predictions[mask] if predictions is not None else None
            synchronized_actuals = actuals[mask] if actuals is not None else None
            
            print(f"[INFO] {module_type}: Synchronized data length: {len(synchronized_timestamps)}")
            
            # Create synchronized module data
            synchronized_module_data = module_data.copy()
            synchronized_module_data['timestamps'] = synchronized_timestamps
            if synchronized_predictions is not None:
                synchronized_module_data['predictions'] = synchronized_predictions
            if synchronized_actuals is not None:
                synchronized_module_data['actuals'] = synchronized_actuals
            
            synchronized_results[module_type] = synchronized_module_data
        
        return synchronized_results

    def _synchronize_data(self, data: pd.DataFrame, common_start: pd.Timestamp, module_type: str) -> pd.DataFrame:
        """
        Synchronize a single dataset to start from common_start
        
        Args:
            data: Dataset to synchronize
            common_start: Common start timestamp
            module_type: Type of module for logging
            
        Returns:
            Synchronized dataset
        """
        print(f"[INFO] {module_type}: Synchronizing data to start from {common_start}")
        
        # Find rows where timestamp >= common_start
        mask = data['timestamp'] >= common_start
        
        if not mask.any():
            print(f"[WARN] {module_type}: No data available after {common_start}")
            return data
        
        # Filter data to start from common_start
        synchronized_data = data[mask].copy()
        
        print(f"[INFO] {module_type}: Original data length: {len(data)}")
        print(f"[INFO] {module_type}: Synchronized data length: {len(synchronized_data)}")
        print(f"[INFO] {module_type}: New start time: {synchronized_data['timestamp'].min()}")
        print(f"[INFO] {module_type}: New end time: {synchronized_data['timestamp'].max()}")
        
        return synchronized_data

    def _filter_excluded_periods(self, data: pd.DataFrame, module_type: str = "unknown") -> pd.DataFrame:
        """
        Filter out excluded time periods from the dataset
        
        Args:
            data: DataFrame with timestamp column
            module_type: Type of module for logging purposes
            
        Returns:
            Filtered DataFrame with excluded periods removed
        """
        if not self.exclude_periods or data.empty:
            return data
        
        original_length = len(data)
        filtered_data = data.copy()
        
        # Ensure timestamp column exists and is datetime
        if 'timestamp' in filtered_data.columns:
            filtered_data['timestamp'] = pd.to_datetime(filtered_data['timestamp'])
        elif '_time' in filtered_data.columns:
            filtered_data['timestamp'] = pd.to_datetime(filtered_data['_time'])
        else:
            print(f"[WARN] No timestamp column found in {module_type} data, cannot filter excluded periods")
            return data
        
        # Create mask for all data points (initially all True)
        mask = pd.Series([True] * len(filtered_data), index=filtered_data.index)
        
        # Apply exclusions
        for start_date_str, end_date_str in self.exclude_periods:
            try:
                start_date = pd.to_datetime(start_date_str)
                end_date = pd.to_datetime(end_date_str)
                
                # Create exclusion mask for this period
                period_mask = (filtered_data['timestamp'] >= start_date) & (filtered_data['timestamp'] <= end_date)
                
                # Update overall mask (exclude this period)
                mask = mask & ~period_mask
                
                excluded_count = period_mask.sum()
                if excluded_count > 0:
                    print(f"[INFO] {module_type.upper()}: Excluded {excluded_count} records from {start_date_str} to {end_date_str}")
                
            except Exception as e:
                print(f"[WARN] Could not parse exclusion period {start_date_str} to {end_date_str}: {e}")
                continue
        
        # Apply the mask
        filtered_data = filtered_data[mask].reset_index(drop=True)
        
        excluded_count = original_length - len(filtered_data)
        if excluded_count > 0:
            print(f"[INFO] {module_type.upper()}: Filtered out {excluded_count} records ({excluded_count/original_length*100:.1f}%) from {original_length} total records")
            print(f"[INFO] {module_type.upper()}: Remaining data: {len(filtered_data)} records from {filtered_data['timestamp'].min()} to {filtered_data['timestamp'].max()}")
        else:
            print(f"[INFO] {module_type.upper()}: No records were excluded")
        
        return filtered_data

    def run_analytical_packages(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run analytical packages for in-depth analysis
        
        Args:
            evaluation_results: Results from standard evaluation
            
        Returns:
            Analytical packages results
        """
        if not ANALYTICAL_PACKAGES_AVAILABLE:
            print("[WARN] Analytical packages not available, skipping analytical analysis")
            return {}
        
        print("="*60)
        print("ANALYTICAL PACKAGES ANALYSIS")
        print("="*60)
        
        analytical_results = {}
        
        for module_type, module_data in evaluation_results.items():
            if module_type in ['silicon', 'perovskite']:
                print(f"\n[INFO] Running analytical packages for {module_type} modules...")
                
                # Extract data for analytical analysis
                predictions = module_data.get('predictions')
                actuals = module_data.get('actuals')
                timestamps = module_data.get('timestamps')
                
                if predictions is None or actuals is None or timestamps is None:
                    print(f"[WARN] Missing data for {module_type} analytical analysis")
                    continue
                
                # Run Analytical Package 1: Variability Analysis
                try:
                    print(f"[INFO] Running Package 1: Variability Analysis for {module_type}")
                    cv_results = analyze_cv_prediction_quality(
                        predictions=predictions,
                        actuals=actuals,
                        timestamps=timestamps,
                        module_type=module_type
                    )
                    
                    analytical_results[f"{module_type}_variability"] = cv_results
                    
                    # Create visualizations
                    if cv_results:
                        analyzer = VariabilityAnalysis(module_type)
                        figures = analyzer.create_cv_visualizations(cv_results)
                        analytical_results[f"{module_type}_variability_figures"] = figures
                    
                except Exception as e:
                    print(f"[ERROR] Package 1 failed for {module_type}: {e}")
                
                # Run Analytical Package 2: Ramp-Rate Analysis
                try:
                    print(f"[INFO] Running Package 2: Ramp-Rate Analysis for {module_type}")
                    ramp_results = analyze_ramp_rate_prediction_quality(
                        predictions=predictions,
                        actuals=actuals,
                        timestamps=timestamps,
                        module_type=module_type
                    )
                    
                    analytical_results[f"{module_type}_ramp_rate"] = ramp_results
                    
                    # Create visualizations
                    if ramp_results:
                        analyzer = RampRateAnalysis(module_type)
                        figures = analyzer.create_ramp_rate_visualizations(ramp_results)
                        analytical_results[f"{module_type}_ramp_rate_figures"] = figures
                    
                except Exception as e:
                    print(f"[ERROR] Package 2 failed for {module_type}: {e}")
                
                # Run Analytical Package 3: Intermittency Analysis
                try:
                    print(f"[INFO] Running Package 3: Intermittency Analysis for {module_type}")
                    intermittency_results = analyze_intermittency_prediction_quality(
                        predictions=predictions,
                        actuals=actuals,
                        timestamps=timestamps,
                        module_type=module_type
                    )
                    
                    analytical_results[f"{module_type}_intermittency"] = intermittency_results
                    
                    # Create visualizations
                    if intermittency_results:
                        analyzer = IntermittencyAnalysis(module_type)
                        figures = analyzer.create_intermittency_visualizations(intermittency_results)
                        analytical_results[f"{module_type}_intermittency_figures"] = figures
                    
                except Exception as e:
                    print(f"[ERROR] Package 3 failed for {module_type}: {e}")
                
                # Run Analytical Package 4: Scatter Plot Region Analysis
                try:
                    print(f"[INFO] Running Package 4: Scatter Plot Region Analysis for {module_type}")
                    from analytical_packages.package4_scatter_region_analysis import analyze_scatter_plot_regions
                    
                    scatter_region_results = analyze_scatter_plot_regions(
                        predictions=predictions,
                        actuals=actuals,
                        timestamps=timestamps,
                        module_type=module_type
                    )
                    
                    analytical_results[f"{module_type}_scatter_regions"] = scatter_region_results
                    
                    # Save results to file for later analysis
                    if hasattr(self, 'output_dir'):
                        from analytical_packages.package4_scatter_region_analysis import ScatterRegionAnalysis
                        analyzer = ScatterRegionAnalysis(module_type)
                        results_file = analyzer.save_results_to_file(scatter_region_results, self.output_dir)
                        analytical_results[f"{module_type}_scatter_regions_file"] = results_file
                    
                except Exception as e:
                    print(f"[ERROR] Package 4 failed for {module_type}: {e}")
                
                # TODO: Add more analytical packages here
                # Package 5: Heteroskedasticity Check
        
        self.analytical_results = analytical_results
        return analytical_results

    
    def run_standard_evaluation(self) -> Dict[str, Any]:
        """Performs standard evaluation for both module types using appropriate evaluator"""
        print("="*60)
        print("STANDARD EVALUATION")
        print("="*60)
        
        # Ensure model is loaded before evaluation
        if self.preprocessor is None:
            print("[INFO] Model not loaded yet, loading now...")
            self.load_model()
        
        # Load weather-integrated data for both module types
        features = list(self.preprocessor.feature_scalers.keys()) + list(self.preprocessor.output_scalers.keys())
        
        print(f"[INFO] Features being loaded:")
        print(f"  - Feature scalers (INPUT): {list(self.preprocessor.feature_scalers.keys())}")
        print(f"  - Output scalers (OUTPUT): {list(self.preprocessor.output_scalers.keys())}")
        print(f"  - Combined features: {features}")
        
        si_data, pvk_data = self.load_both_module_data(features)
        
        # Synchronize data before evaluation if both datasets are available
        if si_data is not None and pvk_data is not None:
            print("\n" + "="*40)
            print("SYNCHRONIZING DATA BEFORE EVALUATION")
            print("="*40)
            
            # Find common start time
            common_start = self._find_common_start_time(si_data, pvk_data)
            
            # Synchronize both datasets to start from common_start
            si_data = self._synchronize_data(si_data, common_start, "Silicon")
            pvk_data = self._synchronize_data(pvk_data, common_start, "Perovskite")
        
        results = {}
        
        # Evaluate Silicon data
        if si_data is not None:
            print("\n" + "="*40)
            print("EVALUATING SILICON MODULES")
            print("="*40)
            si_results = self._evaluate_module_type(si_data, "Silicon")
            results["silicon"] = si_results
        
        # Evaluate Perovskite data
        if pvk_data is not None:
            print("\n" + "="*40)
            print("EVALUATING PEROVSKITE MODULES")
            print("="*40)
            pvk_results = self._evaluate_module_type(pvk_data, "Perovskite")
            results["perovskite"] = pvk_results
        
        return results
    
    def _evaluate_module_type(self, test_data: pd.DataFrame, module_name: str) -> Dict[str, Any]:
        """Evaluate a specific module type using the appropriate evaluator"""
        if self.evaluator is None:
            raise ValueError("No evaluator available. Call load_model() first.")
        
        # Store timestamps for visualization (full series from data file)
        self.timestamps = test_data['timestamp']
        
        # Use the appropriate evaluator
        return self.evaluator.evaluate_module_type(test_data, module_name)
    
    
    
    def create_visualizations_html(self, evaluation_results=None, analytical_results=None):
        """Erzeugt Plotly-Visualisierungen als HTML-DIV-Strings (keine Dateien mehr)"""
        from plotly.io import to_html
        output_features = self.config["training_settings"]["output"]
        plot_divs = {}
        
        # Use provided evaluation results or fall back to old format
        if evaluation_results is None:
            # Fallback to old format for backward compatibility
            if hasattr(self, 'predictions') and hasattr(self, 'actuals'):
                evaluation_results = {
                    'silicon': {
                        'predictions': self.predictions,
                        'actuals': self.actuals,
                        'timestamps': getattr(self, 'eval_timestamps', None)
                    }
                }
            else:
                print("[WARN] No evaluation results provided and no old data available")
                return {}

        # 1. Scatter Plot - Separate plots for Silicon and Perovskite
        for module_type, module_data in evaluation_results.items():
            if module_data is None or 'predictions' not in module_data or 'actuals' not in module_data:
                continue
                
            predictions = module_data['predictions']
            actuals = module_data['actuals']
            
            # Handle multi-step predictions: flatten if needed
            if len(predictions.shape) == 3:  # Multi-step: (n_sequences, forecast_steps, n_features)
                print(f"[INFO] Flattening multi-step predictions for scatter plot: {predictions.shape}")
                predictions_flat = predictions.reshape(-1, predictions.shape[-1])
                actuals_flat = actuals.reshape(-1, actuals.shape[-1])
            else:  # One-step: (n_points, n_features)
                predictions_flat = predictions
                actuals_flat = actuals
            
            # Create descriptive titles for P_normalized
            subplot_titles = []
            for feature in output_features:
                if feature == "P_normalized":
                    subplot_titles.append(f"{module_type.upper()} {feature} - Predictions vs. Actual Values (Quantile-Scaled 0-1)")
                else:
                    subplot_titles.append(f"{module_type.upper()} {feature} - Predictions vs. Actual Values")
            
            fig_scatter = make_subplots(
                rows=len(output_features), cols=1,
                subplot_titles=subplot_titles,
                vertical_spacing=0.1
            )
            for i, feature in enumerate(output_features):
                fig_scatter.add_trace(
                    go.Scatter(
                        x=actuals_flat[:, i],
                        y=predictions_flat[:, i],
                        mode='markers',
                        name=f"{module_type.upper()} {feature}",
                        marker=dict(size=3, opacity=0.6)
                    ),
                    row=i+1, col=1
                )
                min_val = min(actuals_flat[:, i].min(), predictions_flat[:, i].min())
                max_val = max(actuals_flat[:, i].max(), predictions_flat[:, i].max())
                fig_scatter.add_trace(
                    go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        name=f"{module_type.upper()} {feature} Perfect",
                        line=dict(dash='dash', color='red'),
                        showlegend=False
                    ),
                    row=i+1, col=1
                )
            fig_scatter.update_layout(height=300*len(output_features), title=f"{module_type.upper()} Predictions vs. Actual Values")
            plot_divs[f'scatter_{module_type}'] = to_html(fig_scatter, include_plotlyjs='cdn', full_html=False)

        # 2. Time Series + Weather subplot - Combined view with both module types
        if evaluation_results:
            # Create combined time series plot showing both Silicon and Perovskite
            rows = 2
            main_title = f"{output_features[0]} - Time Series Comparison (Silicon vs Perovskite)"
            if output_features[0] == "P_normalized":
                main_title = f"{output_features[0]} - Time Series Comparison (Quantile-Scaled 0-1) - Silicon vs Perovskite"
            
            fig_time = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.12,
                subplot_titles=[main_title, "P_normalized - Difference Analysis - Silicon vs Perovskite", "Weather Features"]
            )
            
            # Get timestamps from first available module type
            timestamps = None
            for module_type, module_data in evaluation_results.items():
                if module_data and 'timestamps' in module_data and module_data['timestamps'] is not None:
                    timestamps = module_data['timestamps']
                    break
            
            if timestamps is None:
                # Fallback to generated timestamps
                max_length = max([len(module_data['actuals']) for module_data in evaluation_results.values() if module_data and 'actuals' in module_data])
                timestamps = pd.date_range(start=datetime.now(), periods=max_length, freq='5min')
            
            # Handle multi-step timestamps: ensure they match flattened predictions
            if timestamps is not None and len(timestamps) > 0:
                # Check if we need to flatten timestamps for multi-step predictions
                for module_type, module_data in evaluation_results.items():
                    if module_data and 'predictions' in module_data:
                        predictions = module_data['predictions']
                        if len(predictions.shape) == 3:  # Multi-step: (n_sequences, forecast_steps, n_features)
                            print(f"[INFO] Using pre-flattened timestamps for multi-step visualization: {len(timestamps)} timestamps")
                            # For multi-step, timestamps are already flattened in the evaluator
                            # No need to modify them here
                            break
            
            # Power comparison for both module types
            for module_type, module_data in evaluation_results.items():
                if module_data is None or 'predictions' not in module_data or 'actuals' not in module_data:
                    continue
                    
                predictions = module_data['predictions']
                actuals = module_data['actuals']
                
                # Get timestamps for this specific module type
                module_timestamps = module_data.get('timestamps', timestamps)
                if module_timestamps is None:
                    module_timestamps = timestamps
                
                # Handle multi-step predictions: flatten if needed
                if len(predictions.shape) == 3:  # Multi-step: (n_sequences, forecast_steps, n_features)
                    print(f"[INFO] Flattening multi-step predictions for visualization: {predictions.shape}")
                    predictions_flat = predictions.reshape(-1, predictions.shape[-1])
                    actuals_flat = actuals.reshape(-1, actuals.shape[-1])
                else:  # One-step: (n_points, n_features)
                    predictions_flat = predictions
                    actuals_flat = actuals
                
                for i, feature in enumerate(output_features):
                    # Actual values
                    fig_time.add_trace(go.Scatter(
                        x=module_timestamps, 
                        y=actuals_flat[:, i], 
                        mode='lines',
                        name=f"{module_type.upper()} {feature} Actual", 
                        line=dict(color='blue' if module_type == 'silicon' else 'green', width=2)
                    ), row=1, col=1)
                    
                    # Predicted values
                    fig_time.add_trace(go.Scatter(
                        x=module_timestamps, 
                        y=predictions_flat[:, i], 
                        mode='lines',
                        name=f"{module_type.upper()} {feature} Predicted", 
                        line=dict(color='red' if module_type == 'silicon' else 'orange', width=1, dash='dash')
                    ), row=1, col=1)
            # Weather features, if available
            weather_features = self.config.get("weather_data", {}).get("weather_features", [])
            
            if weather_features:
                wf_colors = {'TT_10':'orange','RF_10':'purple','V_N':'brown','RWS_10':'cyan','RWS_IND_10':'pink','GS_10':'yellow','SD_10':'gray','Irr':'green'}
                # Versuche die im Evaluationsfenster genutzten Wetterwerte zu zeichnen
                try:
                    # Load weather data for visualization - both Silicon and Perovskite
                    feats = list(self.preprocessor.feature_scalers.keys()) + list(self.preprocessor.output_scalers.keys())
                    
                    # Load Silicon weather data
                    df_w_si_original = self.load_weather_integrated_data(feats, module_type="silicon")
                    df_w_si_original = df_w_si_original.sort_values('timestamp')
                    
                    # Load Perovskite weather data
                    df_w_pvk_original = self.load_weather_integrated_data(feats, module_type="perovskite")
                    df_w_pvk_original = df_w_pvk_original.sort_values('timestamp')
                    
                    # Use entire dataset for weather visualization (no time filtering)
                    print(f"[INFO] Using entire weather datasets for visualization")
                    
                    # Create aligned versions for Irr plotting (only for predictions)
                    try:
                        tmin, tmax = timestamps.iloc[0], timestamps.iloc[-1]
                    except Exception:
                        tmin, tmax = timestamps[0], timestamps[-1]
                    
                    df_w_si = df_w_si_original[(df_w_si_original['timestamp'] >= tmin) & (df_w_si_original['timestamp'] <= tmax)].copy()
                    df_w_si = df_w_si.set_index('timestamp').reindex(timestamps)
                    
                    df_w_pvk = df_w_pvk_original[(df_w_pvk_original['timestamp'] >= tmin) & (df_w_pvk_original['timestamp'] <= tmax)].copy()
                    df_w_pvk = df_w_pvk.set_index('timestamp').reindex(timestamps)
                    
                    # Add Irr values for both module types
                    if 'Irr' in df_w_si.columns and 'Irr' in df_w_pvk.columns:
                        # Silicon Irr
                        fig_time.add_trace(
                            go.Scatter(x=timestamps, y=df_w_si['Irr'].values, mode='lines',
                                       name='Irr (Silicon)', line=dict(color='blue', width=2), opacity=0.8),
                            row=3, col=1)
                        
                        # Perovskite Irr
                        fig_time.add_trace(
                            go.Scatter(x=timestamps, y=df_w_pvk['Irr'].values, mode='lines',
                                       name='Irr (Perovskite)', line=dict(color='green', width=2), opacity=0.8),
                            row=3, col=1)
                        
                        print(f"[INFO] Added Irr values for both Silicon and Perovskite modules to weather subplot")
                    
                    # Add other weather features (use original Silicon data for full dataset)
                    df_w = df_w_si_original  # Use original Silicon data for weather features
                    for feat in weather_features:
                        if feat != 'Irr' and feat in df_w.columns:  # Skip Irr as it's handled separately above
                            # Use full weather dataset timestamps, not just prediction timestamps
                            weather_timestamps = df_w['timestamp'].tolist()
                            fig_time.add_trace(
                                go.Scatter(x=weather_timestamps, y=df_w[feat].values, mode='lines',
                                           name=feat, line=dict(color=wf_colors.get(feat,'black'), width=2), opacity=0.8),
                                row=3, col=1)
                            print(f"[INFO] Added weather feature '{feat}' with {len(weather_timestamps)} timestamps")
                    
                    print(f"[INFO] Weather features plotted for full dataset: {len(weather_timestamps)} timestamps")
                except Exception as e:
                    print(f"[WARN] Could not add weather features to subplot: {e}")
            
            # Add Difference Analysis (Row 2)
            try:
                # Get Silicon and Perovskite data for comparison
                si_data = evaluation_results.get('silicon', {})
                pvk_data = evaluation_results.get('perovskite', {})
                
                if si_data and pvk_data and 'actuals' in si_data and 'actuals' in pvk_data and 'predictions' in si_data and 'predictions' in pvk_data:
                    si_actuals = si_data['actuals']
                    si_predictions = si_data['predictions']
                    pvk_actuals = pvk_data['actuals']
                    pvk_predictions = pvk_data['predictions']
                    
                    # Ensure all arrays have the same length
                    min_length = min(len(si_actuals), len(pvk_actuals), len(si_predictions), len(pvk_predictions))
                    si_actuals = si_actuals[:min_length]
                    si_predictions = si_predictions[:min_length]
                    pvk_actuals = pvk_actuals[:min_length]
                    pvk_predictions = pvk_predictions[:min_length]
                    
                    # Calculate differences
                    actual_diff_si_pvk = si_actuals[:, 0] - pvk_actuals[:, 0]  # Si actual - Pvk actual
                    pred_error_si = si_predictions[:, 0] - si_actuals[:, 0]    # Si prediction - Si actual
                    pred_error_pvk = pvk_predictions[:, 0] - pvk_actuals[:, 0] # Pvk prediction - Pvk actual
                    pred_diff_si_pvk = si_predictions[:, 0] - pvk_predictions[:, 0] # Si prediction - Pvk prediction
                    
                    # Use timestamps from Silicon data
                    diff_timestamps = si_data.get('timestamps', timestamps)
                    if diff_timestamps is None:
                        diff_timestamps = timestamps
                    
                    # Ensure timestamps match data length
                    if len(diff_timestamps) > min_length:
                        diff_timestamps = diff_timestamps[:min_length]
                    
                    # Add difference traces
                    fig_time.add_trace(
                        go.Scatter(x=diff_timestamps, y=actual_diff_si_pvk, mode='lines',
                                   name='P_normalized Actual Diff (Si - Pvk)', 
                                   line=dict(color='purple', width=2), opacity=0.8),
                        row=2, col=1)
                    
                    fig_time.add_trace(
                        go.Scatter(x=diff_timestamps, y=pred_error_si, mode='lines',
                                   name='P_normalized Prediction Error (Si)',
                                   line=dict(color='red', width=2, dash='dash'), opacity=0.8),
                        row=2, col=1)
                    
                    fig_time.add_trace(
                        go.Scatter(x=diff_timestamps, y=pred_error_pvk, mode='lines',
                                   name='P_normalized Prediction Error (Pvk)',
                                   line=dict(color='orange', width=2, dash='dash'), opacity=0.8),
                        row=2, col=1)
                    
                    fig_time.add_trace(
                        go.Scatter(x=diff_timestamps, y=pred_diff_si_pvk, mode='lines',
                                   name='P_normalized Prediction Diff (Si - Pvk)',
                                   line=dict(color='brown', width=2, dash='dot'), opacity=0.8),
                        row=2, col=1)
                    
                    # Add Prediction Error Diff (Si - Pvk)
                    pred_error_diff_si_pvk = pred_error_si - pred_error_pvk
                    fig_time.add_trace(
                        go.Scatter(x=diff_timestamps, y=pred_error_diff_si_pvk, mode='lines',
                                   name='P_normalized Prediction Error Diff (Si - Pvk)',
                                   line=dict(color='green', width=2, dash='dash'), opacity=0.8),
                        row=2, col=1)
                    
                    # Add zero line for reference
                    fig_time.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)
                    
                    print(f"[INFO] Added difference analysis with {len(diff_timestamps)} data points")
                    print(f"[INFO] Actual diff (Si-Pvk) range: {actual_diff_si_pvk.min():.4f} to {actual_diff_si_pvk.max():.4f}")
                    print(f"[INFO] Si prediction error range: {pred_error_si.min():.4f} to {pred_error_si.max():.4f}")
                    print(f"[INFO] Pvk prediction error range: {pred_error_pvk.min():.4f} to {pred_error_pvk.max():.4f}")
                    print(f"[INFO] Prediction diff (Si-Pvk) range: {pred_diff_si_pvk.min():.4f} to {pred_diff_si_pvk.max():.4f}")
                    print(f"[INFO] Prediction error diff (Si-Pvk) range: {pred_error_diff_si_pvk.min():.4f} to {pred_error_diff_si_pvk.max():.4f}")
                    
                else:
                    print(f"[WARN] Insufficient data for difference analysis")
                    
            except Exception as e:
                print(f"[WARN] Could not add difference analysis: {e}")
                import traceback
                traceback.print_exc()
            
            fig_time.update_layout(height=1000, title="Time Series Comparison (Silicon vs Perovskite)")
            plot_divs['time_series'] = to_html(fig_time, include_plotlyjs=False, full_html=False)

        # 3. Error Distribution - Separate plots for Silicon and Perovskite
        for module_type, module_data in evaluation_results.items():
            if module_data is None or 'predictions' not in module_data or 'actuals' not in module_data:
                continue
                
            predictions = module_data['predictions']
            actuals = module_data['actuals']
            
            # Create descriptive titles for P_normalized
            error_titles = []
            for feature in output_features:
                if feature == "P_normalized":
                    error_titles.append(f"{module_type.upper()} {feature} - Error Distribution (Quantile-Scaled 0-1)")
                else:
                    error_titles.append(f"{module_type.upper()} {feature} - Error Distribution")
            
            fig_error = make_subplots(
                rows=len(output_features), cols=1,
                subplot_titles=error_titles,
                vertical_spacing=0.1
            )
            for i, feature in enumerate(output_features):
                errors = predictions[:, i] - actuals[:, i]
                fig_error.add_trace(
                    go.Histogram(
                        x=errors,
                        name=f"{module_type.upper()} {feature}",
                        nbinsx=50
                    ),
                    row=i+1, col=1
                )
            fig_error.update_layout(height=300*len(output_features), title=f"{module_type.upper()} Error Distribution")
            plot_divs[f'error_dist_{module_type}'] = to_html(fig_error, include_plotlyjs=False, full_html=False)

        # Add analytical packages visualizations
        analytical_plot_divs = self._create_analytical_plot_divs(analytical_results)
        plot_divs.update(analytical_plot_divs)
        
        return plot_divs
    
    def _create_analytical_plot_divs(self, analytical_results: Dict[str, Any]) -> Dict[str, str]:
        """Create plotly HTML divs for analytical packages with combined Si+Pvk plots"""
        from plotly.io import to_html
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        plot_divs = {}
        
        if not analytical_results:
            print("[DEBUG] No analytical results provided to create plot divs")
            return plot_divs
        
        print(f"[DEBUG] Available analytical results: {list(analytical_results.keys())}")
        
        # Create unified plots for each package (combining Si + Pvk)
        packages = ['variability', 'ramp_rate', 'intermittency', 'power_level']
        
        for package_name in packages:
            print(f"[DEBUG] Creating unified plots for package: {package_name}")
            
            # Get data for both module types
            si_data = analytical_results.get(f'silicon_{package_name}', {})
            pvk_data = analytical_results.get(f'perovskite_{package_name}', {})
            
            if si_data or pvk_data:
                try:
                    # Create 3 unified plots for this package
                    unified_plots = self._create_unified_package_plots(package_name, si_data, pvk_data)
                    
                    if unified_plots:
                        # Combine the 3 plots into a single HTML div
                        combined_html = self._combine_plotly_figures(unified_plots, f"Combined {package_name.title()} Analysis")
                        plot_divs[f'analytical_{package_name}_combined'] = combined_html
                        print(f"[DEBUG] Created unified plots for {package_name}")
                    else:
                        print(f"[DEBUG] No unified plots created for {package_name}")
                        
                except Exception as e:
                    print(f"[ERROR] Failed to create unified plots for {package_name}: {e}")
        
        print(f"[DEBUG] Created unified plot divs: {list(plot_divs.keys())}")
        return plot_divs
    
    def _create_unified_package_plots(self, package_name: str, si_data: Dict, pvk_data: Dict):
        """Create 3 unified plots for a package: Daily Comparison, Error Timeline, Correlation Scatter"""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        plots = []
        
        # Define consistent colors for Si and Pvk (same as time series)
        si_color = '#1f77b4'  # Blue
        pvk_color = '#ff7f0e'  # Orange
        
        try:
            if package_name == 'variability':
                plots = self._create_variability_unified_plots(si_data, pvk_data, si_color, pvk_color)
            elif package_name == 'ramp_rate':
                plots = self._create_ramp_rate_unified_plots(si_data, pvk_data, si_color, pvk_color)
            elif package_name == 'intermittency':
                plots = self._create_intermittency_unified_plots(si_data, pvk_data, si_color, pvk_color)
            elif package_name == 'power_level':
                plots = self._create_power_level_unified_plots(si_data, pvk_data, si_color, pvk_color)
                
        except Exception as e:
            print(f"[ERROR] Failed to create unified plots for {package_name}: {e}")
            
        return plots
    
    def _create_variability_unified_plots(self, si_data: Dict, pvk_data: Dict, si_color: str, pvk_color: str):
        """Create 3 unified plots for variability analysis"""
        import plotly.graph_objects as go
        
        plots = []
        
        try:
            # Plot 1: Daily CV Comparison (Si + Pvk Actuals and Predictions)
            fig1 = go.Figure()
            
            # Add Silicon data
            if si_data and 'daily_data' in si_data:
                si_daily = si_data['daily_data']
                fig1.add_trace(go.Scatter(
                    x=si_daily['date'] if 'date' in si_daily else si_daily.index,
                    y=si_daily['actual_cv'],
                    mode='lines+markers',
                    name='Silicon Actual CV',
                    line=dict(color=si_color, width=2),
                    marker=dict(size=4)
                ))
                fig1.add_trace(go.Scatter(
                    x=si_daily['date'] if 'date' in si_daily else si_daily.index,
                    y=si_daily['predicted_cv'],
                    mode='lines+markers',
                    name='Silicon Predicted CV',
                    line=dict(color=si_color, width=2, dash='dash'),
                    marker=dict(size=4, symbol='square')
                ))
            
            # Add Perovskite data
            if pvk_data and 'daily_data' in pvk_data:
                pvk_daily = pvk_data['daily_data']
                fig1.add_trace(go.Scatter(
                    x=pvk_daily['date'] if 'date' in pvk_daily else pvk_daily.index,
                    y=pvk_daily['actual_cv'],
                    mode='lines+markers',
                    name='Perovskite Actual CV',
                    line=dict(color=pvk_color, width=2),
                    marker=dict(size=4)
                ))
                fig1.add_trace(go.Scatter(
                    x=pvk_daily['date'] if 'date' in pvk_daily else pvk_daily.index,
                    y=pvk_daily['predicted_cv'],
                    mode='lines+markers',
                    name='Perovskite Predicted CV',
                    line=dict(color=pvk_color, width=2, dash='dash'),
                    marker=dict(size=4, symbol='square')
                ))
            
            fig1.update_layout(
                title='Daily Coefficient of Variation - Actual vs Predicted',
                xaxis_title='Date',
                yaxis_title='Coefficient of Variation',
                showlegend=True,
                template='plotly_white'
            )
            plots.append(('daily_cv_comparison', fig1))
            
            # Plot 2: CV Prediction Error Timeline
            fig2 = go.Figure()
            
            # Add Silicon CV errors
            if si_data and 'daily_data' in si_data:
                si_daily = si_data['daily_data']
                fig2.add_trace(go.Scatter(
                    x=si_daily['date'] if 'date' in si_daily else si_daily.index,
                    y=si_daily['cv_error_absolute'],
                    mode='lines+markers',
                    name='Silicon CV Error',
                    line=dict(color=si_color, width=2),
                    marker=dict(size=4)
                ))
            
            # Add Perovskite CV errors
            if pvk_data and 'daily_data' in pvk_data:
                pvk_daily = pvk_data['daily_data']
                fig2.add_trace(go.Scatter(
                    x=pvk_daily['date'] if 'date' in pvk_daily else pvk_daily.index,
                    y=pvk_daily['cv_error_absolute'],
                    mode='lines+markers',
                    name='Perovskite CV Error',
                    line=dict(color=pvk_color, width=2),
                    marker=dict(size=4)
                ))
            
            fig2.update_layout(
                title='CV Prediction Error Timeline',
                xaxis_title='Date',
                yaxis_title='Absolute CV Error',
                showlegend=True,
                template='plotly_white'
            )
            plots.append(('cv_error_timeline', fig2))
            
            # Plot 3: CV Correlation Scatter (Si vs Pvk)
            fig3 = go.Figure()
            
            # Add Silicon scatter
            if si_data and 'daily_data' in si_data:
                si_daily = si_data['daily_data']
                fig3.add_trace(go.Scatter(
                    x=si_daily['actual_cv'],
                    y=si_daily['predicted_cv'],
                    mode='markers',
                    name='Silicon',
                    marker=dict(color=si_color, size=6, opacity=0.7)
                ))
            
            # Add Perovskite scatter
            if pvk_data and 'daily_data' in pvk_data:
                pvk_daily = pvk_data['daily_data']
                fig3.add_trace(go.Scatter(
                    x=pvk_daily['actual_cv'],
                    y=pvk_daily['predicted_cv'],
                    mode='markers',
                    name='Perovskite',
                    marker=dict(color=pvk_color, size=6, opacity=0.7)
                ))
            
            # Add perfect prediction line
            if (si_data and 'daily_data' in si_data) or (pvk_data and 'daily_data' in pvk_data):
                # Determine range for perfect prediction line
                all_actuals = []
                if si_data and 'daily_data' in si_data:
                    all_actuals.extend(si_data['daily_data']['actual_cv'].tolist())
                if pvk_data and 'daily_data' in pvk_data:
                    all_actuals.extend(pvk_data['daily_data']['actual_cv'].tolist())
                
                if all_actuals:
                    min_val = min(all_actuals)
                    max_val = max(all_actuals)
                    fig3.add_trace(go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        name='Perfect Prediction',
                        line=dict(color='red', width=2, dash='dash'),
                        showlegend=True
                    ))
            
            fig3.update_layout(
                title='CV Prediction Correlation - Silicon vs Perovskite',
                xaxis_title='Actual CV',
                yaxis_title='Predicted CV',
                showlegend=True,
                template='plotly_white'
            )
            plots.append(('cv_correlation_scatter', fig3))
            
        except Exception as e:
            print(f"[ERROR] Failed to create variability plots: {e}")
        
        return plots
    
    def _create_ramp_rate_unified_plots(self, si_data: Dict, pvk_data: Dict, si_color: str, pvk_color: str):
        """Create 3 unified plots for ramp-rate analysis"""
        import plotly.graph_objects as go
        
        plots = []
        
        try:
            # Plot 1: Daily Ramp Rate Comparison (Si + Pvk Actuals and Predictions)
            fig1 = go.Figure()
            
            # Add Silicon data
            if si_data and 'daily_data' in si_data:
                si_daily = si_data['daily_data']
                fig1.add_trace(go.Scatter(
                    x=si_daily['date'] if 'date' in si_daily else si_daily.index,
                    y=si_daily['actual_mean_ramp'],
                    mode='lines+markers',
                    name='Silicon Actual Ramp',
                    line=dict(color=si_color, width=2),
                    marker=dict(size=4)
                ))
                fig1.add_trace(go.Scatter(
                    x=si_daily['date'] if 'date' in si_daily else si_daily.index,
                    y=si_daily['predicted_mean_ramp'],
                    mode='lines+markers',
                    name='Silicon Predicted Ramp',
                    line=dict(color=si_color, width=2, dash='dash'),
                    marker=dict(size=4, symbol='square')
                ))
            
            # Add Perovskite data
            if pvk_data and 'daily_data' in pvk_data:
                pvk_daily = pvk_data['daily_data']
                fig1.add_trace(go.Scatter(
                    x=pvk_daily['date'] if 'date' in pvk_daily else pvk_daily.index,
                    y=pvk_daily['actual_mean_ramp'],
                    mode='lines+markers',
                    name='Perovskite Actual Ramp',
                    line=dict(color=pvk_color, width=2),
                    marker=dict(size=4)
                ))
                fig1.add_trace(go.Scatter(
                    x=pvk_daily['date'] if 'date' in pvk_daily else pvk_daily.index,
                    y=pvk_daily['predicted_mean_ramp'],
                    mode='lines+markers',
                    name='Perovskite Predicted Ramp',
                    line=dict(color=pvk_color, width=2, dash='dash'),
                    marker=dict(size=4, symbol='square')
                ))
            
            fig1.update_layout(
                title='Daily Ramp Rate - Actual vs Predicted',
                xaxis_title='Date',
                yaxis_title='Mean Ramp Rate',
                showlegend=True,
                template='plotly_white'
            )
            plots.append(('daily_ramp_comparison', fig1))
            
            # Plot 2: Ramp Rate Prediction Error Timeline
            fig2 = go.Figure()
            
            # Add Silicon ramp errors
            if si_data and 'daily_data' in si_data:
                si_daily = si_data['daily_data']
                fig2.add_trace(go.Scatter(
                    x=si_daily['date'] if 'date' in si_daily else si_daily.index,
                    y=si_daily['ramp_error_mean_abs'],
                    mode='lines+markers',
                    name='Silicon Ramp Error',
                    line=dict(color=si_color, width=2),
                    marker=dict(size=4)
                ))
            
            # Add Perovskite ramp errors
            if pvk_data and 'daily_data' in pvk_data:
                pvk_daily = pvk_data['daily_data']
                fig2.add_trace(go.Scatter(
                    x=pvk_daily['date'] if 'date' in pvk_daily else pvk_daily.index,
                    y=pvk_daily['ramp_error_mean_abs'],
                    mode='lines+markers',
                    name='Perovskite Ramp Error',
                    line=dict(color=pvk_color, width=2),
                    marker=dict(size=4)
                ))
            
            fig2.update_layout(
                title='Ramp Rate Prediction Error Timeline',
                xaxis_title='Date',
                yaxis_title='Absolute Ramp Rate Error',
                showlegend=True,
                template='plotly_white'
            )
            plots.append(('ramp_error_timeline', fig2))
            
            # Plot 3: Ramp Rate Correlation Scatter (Si vs Pvk)
            fig3 = go.Figure()
            
            # Add Silicon scatter
            if si_data and 'daily_data' in si_data:
                si_daily = si_data['daily_data']
                fig3.add_trace(go.Scatter(
                    x=si_daily['actual_mean_ramp'],
                    y=si_daily['predicted_mean_ramp'],
                    mode='markers',
                    name='Silicon',
                    marker=dict(color=si_color, size=6, opacity=0.7)
                ))
            
            # Add Perovskite scatter
            if pvk_data and 'daily_data' in pvk_data:
                pvk_daily = pvk_data['daily_data']
                fig3.add_trace(go.Scatter(
                    x=pvk_daily['actual_mean_ramp'],
                    y=pvk_daily['predicted_mean_ramp'],
                    mode='markers',
                    name='Perovskite',
                    marker=dict(color=pvk_color, size=6, opacity=0.7)
                ))
            
            # Add perfect prediction line
            if (si_data and 'daily_data' in si_data) or (pvk_data and 'daily_data' in pvk_data):
                all_actuals = []
                if si_data and 'daily_data' in si_data:
                    all_actuals.extend(si_data['daily_data']['actual_mean_ramp'].tolist())
                if pvk_data and 'daily_data' in pvk_data:
                    all_actuals.extend(pvk_data['daily_data']['actual_mean_ramp'].tolist())
                
                if all_actuals:
                    min_val = min(all_actuals)
                    max_val = max(all_actuals)
                    fig3.add_trace(go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        name='Perfect Prediction',
                        line=dict(color='red', width=2, dash='dash'),
                        showlegend=True
                    ))
            
            fig3.update_layout(
                title='Ramp Rate Prediction Correlation - Silicon vs Perovskite',
                xaxis_title='Actual Mean Ramp Rate',
                yaxis_title='Predicted Mean Ramp Rate',
                showlegend=True,
                template='plotly_white'
            )
            plots.append(('ramp_correlation_scatter', fig3))
            
        except Exception as e:
            print(f"[ERROR] Failed to create ramp rate plots: {e}")
        
        return plots
    
    def _create_intermittency_unified_plots(self, si_data: Dict, pvk_data: Dict, si_color: str, pvk_color: str):
        """Create 3 unified plots for intermittency analysis"""
        import plotly.graph_objects as go
        
        plots = []
        
        try:
            # Similar structure to above but for intermittency metrics
            # Plot 1: Daily Intermittency Comparison
            fig1 = go.Figure()
            
            if si_data and 'daily_intermittency' in si_data:
                si_daily = si_data['daily_intermittency']
                if hasattr(si_daily, 'index') and len(si_daily) > 0:
                    fig1.add_trace(go.Scatter(
                        x=si_daily['date'] if 'date' in si_daily.columns else si_daily.index,
                        y=si_daily['actual_intermittency_ratio'] if 'actual_intermittency_ratio' in si_daily.columns else [],
                    mode='lines+markers',
                    name='Silicon Actual Intermittency',
                    line=dict(color=si_color, width=2),
                    marker=dict(size=4)
                ))
                    fig1.add_trace(go.Scatter(
                        x=si_daily['date'] if 'date' in si_daily.columns else si_daily.index,
                        y=si_daily['predicted_intermittency_ratio'] if 'predicted_intermittency_ratio' in si_daily.columns else [],
                    mode='lines+markers',
                    name='Silicon Predicted Intermittency',
                    line=dict(color=si_color, width=2, dash='dash'),
                    marker=dict(size=4, symbol='square')
                ))
            
            if pvk_data and 'daily_intermittency' in pvk_data:
                pvk_daily = pvk_data['daily_intermittency']
                if hasattr(pvk_daily, 'index') and len(pvk_daily) > 0:
                    fig1.add_trace(go.Scatter(
                        x=pvk_daily['date'] if 'date' in pvk_daily.columns else pvk_daily.index,
                        y=pvk_daily['actual_intermittency_ratio'] if 'actual_intermittency_ratio' in pvk_daily.columns else [],
                    mode='lines+markers',
                    name='Perovskite Actual Intermittency',
                    line=dict(color=pvk_color, width=2),
                    marker=dict(size=4)
                ))
                fig1.add_trace(go.Scatter(
                        x=pvk_daily['date'] if 'date' in pvk_daily.columns else pvk_daily.index,
                        y=pvk_daily['predicted_intermittency_ratio'] if 'predicted_intermittency_ratio' in pvk_daily.columns else [],
                    mode='lines+markers',
                    name='Perovskite Predicted Intermittency',
                    line=dict(color=pvk_color, width=2, dash='dash'),
                    marker=dict(size=4, symbol='square')
                ))
            
            fig1.update_layout(
                title='Daily Intermittency - Actual vs Predicted',
                xaxis_title='Time Period',
                yaxis_title='Intermittency Rate',
                showlegend=True,
                template='plotly_white'
            )
            plots.append(('daily_intermittency_comparison', fig1))
            
            # Plot 2: Intermittency Error Timeline
            fig2 = go.Figure()
            
            if si_data and 'daily_intermittency' in si_data:
                si_daily = si_data['daily_intermittency']
                if hasattr(si_daily, 'index') and len(si_daily) > 0 and 'intermittency_ratio_error' in si_daily.columns:
                    fig2.add_trace(go.Scatter(
                        x=si_daily['date'] if 'date' in si_daily.columns else si_daily.index,
                        y=si_daily['intermittency_ratio_error'],
                        mode='lines+markers',
                        name='Silicon Intermittency Error',
                        line=dict(color=si_color, width=2),
                        marker=dict(size=4)
                    ))
            
            if pvk_data and 'daily_intermittency' in pvk_data:
                pvk_daily = pvk_data['daily_intermittency']
                if hasattr(pvk_daily, 'index') and len(pvk_daily) > 0 and 'intermittency_ratio_error' in pvk_daily.columns:
                    fig2.add_trace(go.Scatter(
                        x=pvk_daily['date'] if 'date' in pvk_daily.columns else pvk_daily.index,
                        y=pvk_daily['intermittency_ratio_error'],
                        mode='lines+markers',
                        name='Perovskite Intermittency Error',
                        line=dict(color=pvk_color, width=2),
                        marker=dict(size=4)
                    ))
            
            fig2.update_layout(
                title='Intermittency Error Timeline',
                xaxis_title='Date',
                yaxis_title='Intermittency Ratio Error',
                template='plotly_white'
            )
            plots.append(('intermittency_error_timeline', fig2))
            
            # Plot 3: Intermittency Correlation Scatter
            fig3 = go.Figure()
            
            if si_data and 'daily_intermittency' in si_data:
                si_daily = si_data['daily_intermittency']
                if hasattr(si_daily, 'index') and len(si_daily) > 0:
                    fig3.add_trace(go.Scatter(
                        x=si_daily['actual_intermittency_ratio'] if 'actual_intermittency_ratio' in si_daily.columns else [],
                        y=si_daily['predicted_intermittency_ratio'] if 'predicted_intermittency_ratio' in si_daily.columns else [],
                        mode='markers',
                        name='Silicon Intermittency',
                        marker=dict(color=si_color, size=6, opacity=0.7)
                    ))
            
            if pvk_data and 'daily_intermittency' in pvk_data:
                pvk_daily = pvk_data['daily_intermittency']
                if hasattr(pvk_daily, 'index') and len(pvk_daily) > 0:
                    fig3.add_trace(go.Scatter(
                        x=pvk_daily['actual_intermittency_ratio'] if 'actual_intermittency_ratio' in pvk_daily.columns else [],
                        y=pvk_daily['predicted_intermittency_ratio'] if 'predicted_intermittency_ratio' in pvk_daily.columns else [],
                        mode='markers',
                        name='Perovskite Intermittency',
                        marker=dict(color=pvk_color, size=6, opacity=0.7)
                    ))
            
            # Add diagonal line for perfect correlation
            max_val = 1.0
            fig3.add_trace(go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode='lines',
                name='Perfect Correlation',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            fig3.update_layout(
                title='Intermittency Correlation Scatter',
                xaxis_title='Actual Intermittency Ratio',
                yaxis_title='Predicted Intermittency Ratio',
                template='plotly_white'
            )
            plots.append(('intermittency_correlation_scatter', fig3))
            
        except Exception as e:
            print(f"[ERROR] Failed to create intermittency plots: {e}")
        
        return plots
    
    def _create_power_level_unified_plots(self, si_data: Dict, pvk_data: Dict, si_color: str, pvk_color: str):
        """Create 3 unified plots for power-level analysis"""
        import plotly.graph_objects as go
        
        plots = []
        
        try:
            # Plot 1: Power Level Distribution Comparison
            fig1 = go.Figure()
            
            if si_data and 'power_levels' in si_data:
                power_levels = si_data['power_levels']
                levels = list(power_levels.keys())
                si_counts = [power_levels[level].sum() if power_levels[level] is not None else 0 for level in levels]
                
                fig1.add_trace(go.Bar(
                    x=[level.replace('_', ' ').title() for level in levels],
                    y=si_counts,
                    name='Silicon',
                    marker_color=si_color,
                    opacity=0.7
                ))
            
            if pvk_data and 'power_levels' in pvk_data:
                power_levels = pvk_data['power_levels']
                levels = list(power_levels.keys())
                pvk_counts = [power_levels[level].sum() if power_levels[level] is not None else 0 for level in levels]
                
                fig1.add_trace(go.Bar(
                    x=[level.replace('_', ' ').title() for level in levels],
                    y=pvk_counts,
                    name='Perovskite',
                    marker_color=pvk_color,
                    opacity=0.7
                ))
            
            fig1.update_layout(
                title='Power Level Distribution Comparison',
                xaxis_title='Power Level',
                yaxis_title='Number of Data Points',
                showlegend=True,
                template='plotly_white',
                barmode='group'
            )
            plots.append(('power_level_distribution', fig1))
            
            # Placeholder plots 2 and 3 for power level analysis
            fig2 = go.Figure()
            fig2.add_annotation(text="Power Level Error Analysis<br>(To be implemented based on available metrics)", 
                              xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            fig2.update_layout(title='Power Level Error Analysis', template='plotly_white')
            plots.append(('power_level_error_analysis', fig2))
            
            fig3 = go.Figure()
            fig3.add_annotation(text="Power Level Performance Comparison<br>(To be implemented based on available metrics)", 
                              xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            fig3.update_layout(title='Power Level Performance Comparison', template='plotly_white')
            plots.append(('power_level_performance', fig3))
            
        except Exception as e:
            print(f"[ERROR] Failed to create power level plots: {e}")
        
        return plots
    
    
    def _combine_plotly_figures(self, plotly_figs: list, section_title: str) -> str:
        """Combine multiple plotly figures into a single HTML div using subplots"""
        from plotly.io import to_html
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        
        if not plotly_figs:
            return "<p>No plots available</p>"
        
        # Filter out None figures
        valid_figs = [(name, fig) for name, fig in plotly_figs if fig is not None]
        
        if not valid_figs:
            return f"<p>No valid plots available for {section_title}</p>"
        
        try:
            # If only one figure, return it directly
            if len(valid_figs) == 1:
                fig_name, fig = valid_figs[0]
                return to_html(fig, include_plotlyjs=False, full_html=False)
            
            # Create subplot with multiple figures
            fig_titles = [fig_name.replace('_', ' ').title() for fig_name, _ in valid_figs]
            
            # Create subplots - stack vertically
            combined_fig = make_subplots(
                rows=len(valid_figs), cols=1,
                subplot_titles=fig_titles,
                vertical_spacing=0.08
            )
            
            # Add each plotly figure to the combined plot
            for i, (fig_name, fig) in enumerate(valid_figs):
                for trace in fig.data:
                    combined_fig.add_trace(trace, row=i+1, col=1)
            
            # Update layout
            combined_fig.update_layout(
                height=400 * len(valid_figs),
                title=section_title.replace('_', ' ').title(),
                showlegend=True,
                template='plotly_white'
            )
            
            return to_html(combined_fig, include_plotlyjs=False, full_html=False)
            
        except Exception as e:
            print(f"[ERROR] Failed to combine figures for {section_title}: {e}")
            return f"<p>Error creating combined plot: {str(e)}</p>"
    
    def _create_metrics_table(self, metrics, title):
        """Create HTML table for metrics"""
        if not metrics:
            return f"<h3>{title}</h3><p>No metrics available</p>"
        
        try:
            # Convert metrics dict to DataFrame format
            if isinstance(metrics, dict) and all(isinstance(v, (int, float)) for v in metrics.values()):
                # Single metrics dict - convert to DataFrame
                df_metrics = pd.DataFrame([metrics]).T
                df_metrics.columns = ['Value']
            else:
                # Multiple metrics - use as is
                df_metrics = pd.DataFrame(metrics).T
        except ValueError as e:
            if "all scalar values" in str(e):
                # Handle case where all values are scalars but pandas needs an index
                df_metrics = pd.DataFrame([metrics]).T
                df_metrics.columns = ['Value']
            else:
                raise e
        
        table_html = df_metrics.to_html(float_format="{:.4f}".format, border=1, classes="metrics-table")
        return f"<h3>{title}</h3>{table_html}"
    
    def _create_config_table(self):
        """Create a table showing model configuration (from lstm_forecast_prediction_weatherData.py)"""
        if not self.config:
            return None
        
        # Extract key configuration parameters
        training_settings = self.config.get('training_settings', {})
        
        config_data = {
            'Model Name': self.config.get('model_name', self.model_name),
            'Model Type': 'LSTM',
            'Module Type': 'silicon',
            'Sequence Length': self.config.get('sequence_length', 'N/A'),
            'Hidden Size': self.config.get('hidden_size', 'N/A'),
            'Num Layers': self.config.get('num_layers', 'N/A'),
            'Dropout': self.config.get('dropout', 'N/A'),
            'Epochs': training_settings.get('epochs', 'N/A'),
            'Batch Size': training_settings.get('batch_size', 'N/A'),
            'Learning Rate': training_settings.get('learning_rate', 'N/A'),
            'Loss Function': training_settings.get('loss_function', 'N/A'),
            'Shuffle': training_settings.get('shuffle', 'N/A'),
            'Early Stopping Patience': training_settings.get('early_stopping_patience', 'N/A'),
            'Validation Split': training_settings.get('validation_split', 'N/A'),
        }
        
        # Add weather features if available
        if self.config.get("weather_data", {}).get("use_weatherData", False):
            weather_features = self.config["weather_data"].get("weather_features", [])
            config_data['Weather Features'] = ', '.join(weather_features)
        else:
            config_data['Weather Features'] = 'None'
        
        # Add time features
        time_features = training_settings.get("features", [])
        # Filter out weather features to get only time features
        weather_features_list = self.config.get("weather_data", {}).get("weather_features", [])
        time_features_only = [f for f in time_features if f not in weather_features_list]
        config_data['Time Features'] = ', '.join(time_features_only)
        
        # Add output features
        output_features = training_settings.get("output", [])
        config_data['Output Features'] = ', '.join(output_features)
        
        # Add data resolution
        config_data['Data Resolution'] = self.config.get("data_resolution", training_settings.get("data_resolution", "N/A"))
        
        # Add evaluation method information
        if self.evaluator is not None:
            config_data['Evaluation Method'] = self.evaluator.__class__.__name__
            config_data['Evaluation Type'] = getattr(self.evaluator, 'evaluation_method', 'Unknown')
        else:
            config_data['Evaluation Method'] = 'Not Loaded'
            config_data['Evaluation Type'] = 'Unknown'
        
        # Create table with parameters as rows (instead of columns)
        # Each parameter gets its own row for better readability
        parameter_names = list(config_data.keys())
        parameter_values = list(config_data.values())
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['Parameter', 'Value'],
                fill_color='lightblue',
                align='center',
                font=dict(size=12, color='black', family='Arial')
            ),
            cells=dict(
                values=[parameter_names, parameter_values],
                fill_color='white',
                align=['left', 'center'],
                font=dict(size=11, color='black', family='Arial')
            )
        )])
        
        fig.update_layout(
            title=f'Model Configuration - {self.model_name}',
            height=400,  # Increased height for multiple rows
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        return fig

    def _create_analytical_packages_section(self, analytical_results: Dict[str, Any], plot_divs: Dict[str, str] = None) -> str:
        """Create HTML section for analytical packages results"""
        if not analytical_results:
            return ""
        
        html_sections = []
        
        # Header for analytical section
        html_sections.append("""
        <div class="plot-section">
            <h2>Analytical Packages Analysis</h2>
            <p>Detailed analysis of forecast quality for different characteristics and phenomena specific to Silicon vs. Perovskite modules.</p>
        """)
        
        # Package 1: Variability Analysis
        html_sections.append(self._create_variability_analysis_section(analytical_results, plot_divs))
        
        # Package 2: Ramp-Rate Analysis  
        html_sections.append(self._create_ramp_rate_analysis_section(analytical_results, plot_divs))
        
        # Package 3: Intermittency Analysis
        html_sections.append(self._create_intermittency_analysis_section(analytical_results, plot_divs))
        
        # Package 4: Scatter Plot Region Analysis
        html_sections.append(self._create_scatter_region_analysis_section(analytical_results))
        
        # Summary for Si-Pvk Model Transferability
        html_sections.append(self._create_transferability_summary_section(analytical_results))
        
        html_sections.append("</div>")
        
        return "\n".join(html_sections)
    
    def _get_analytical_plots_html(self, package_name: str, plot_divs: Dict[str, str]) -> str:
        """Get HTML for analytical package plots (now combined Si+Pvk plots)"""
        html = ""
        
        # Look for combined plot
        plot_key = f'analytical_{package_name}_combined'
        if plot_key in plot_divs:
            html += f"""
            <h4>{package_name.title()} Analysis - Combined Silicon vs Perovskite</h4>
            <div class="plot-section">
                {plot_divs[plot_key]}
            </div>
            """
        else:
            # Fallback: look for individual plots (backward compatibility)
            for module_type in ['silicon', 'perovskite']:
                plot_key = f'analytical_{package_name}_{module_type}'
                if plot_key in plot_divs:
                    html += f"""
                    <h4>{module_type.title()} Modules - {package_name.title()} Visualizations</h4>
                    <div class="plot-section">
                        {plot_divs[plot_key]}
                    </div>
                    """
        
        return html
    
    def _create_variability_analysis_section(self, analytical_results: Dict[str, Any], plot_divs: Dict[str, str] = None) -> str:
        """Create Variability Analysis section"""
        html = """
        <div class="analytical-package">
            <h3>Package 1: Variability Analysis</h3>
            <p>Analysis of how well the LSTM model predicts daily coefficient of variation (CV) patterns for different module types.</p>
        """
        
        # Silicon Variability Results
        if 'silicon_variability' in analytical_results:
            html += self._format_variability_results('Silicon', analytical_results['silicon_variability'])
        
        # Perovskite Variability Results
        if 'perovskite_variability' in analytical_results:
            html += self._format_variability_results('Perovskite', analytical_results['perovskite_variability'])
        
        # Add plotly figures if available
        if plot_divs:
            html += self._get_analytical_plots_html('variability', plot_divs)
        
        html += "</div>"
        return html
    
    def _create_ramp_rate_analysis_section(self, analytical_results: Dict[str, Any], plot_divs: Dict[str, str] = None) -> str:
        """Create Ramp-Rate Analysis section"""
        html = """
        <div class="analytical-package">
            <h3>Package 2: Ramp-Rate Analysis</h3>
            <p>Analysis of how well the LSTM model predicts rapid power changes (ramp-rates) for different module types.</p>
        """
        
        # Silicon Ramp-Rate Results
        if 'silicon_ramp_rate' in analytical_results:
            html += self._format_ramp_rate_results('Silicon', analytical_results['silicon_ramp_rate'])
        
        # Perovskite Ramp-Rate Results
        if 'perovskite_ramp_rate' in analytical_results:
            html += self._format_ramp_rate_results('Perovskite', analytical_results['perovskite_ramp_rate'])
        
        # Add plotly figures if available
        if plot_divs:
            html += self._get_analytical_plots_html('ramp_rate', plot_divs)
        
        html += "</div>"
        return html
    
    def _create_intermittency_analysis_section(self, analytical_results: Dict[str, Any], plot_divs: Dict[str, str] = None) -> str:
        """Create Intermittency Analysis section"""
        html = """
        <div class="analytical-package">
            <h3>Package 3: Intermittency Analysis</h3>
            <p>Analysis of how well the LSTM model predicts intermittency patterns and cloud-induced fluctuations for different module types.</p>
        """
        
        # Silicon Intermittency Results
        if 'silicon_intermittency' in analytical_results:
            html += self._format_intermittency_results('Silicon', analytical_results['silicon_intermittency'])
        
        # Perovskite Intermittency Results
        if 'perovskite_intermittency' in analytical_results:
            html += self._format_intermittency_results('Perovskite', analytical_results['perovskite_intermittency'])
        
        # Add plotly figures if available
        if plot_divs:
            html += self._get_analytical_plots_html('intermittency', plot_divs)
        
        html += "</div>"
        return html
    
    def _create_scatter_region_analysis_section(self, analytical_results: Dict[str, Any]) -> str:
        """Create Scatter Plot Region Analysis section"""
        html = """
        <div class="analytical-package">
            <h3>Package 4: Scatter Plot Region Analysis</h3>
            <p>Analysis of model performance across different power level regions to assess transferability between module types. This analysis divides the scatter plot into power level regions and evaluates performance metrics for each region.</p>
        """
        
        # Silicon Region Analysis Results
        if 'silicon_scatter_regions' in analytical_results:
            html += self._format_scatter_region_results('Silicon', analytical_results['silicon_scatter_regions'])
        
        # Perovskite Region Analysis Results
        if 'perovskite_scatter_regions' in analytical_results:
            html += self._format_scatter_region_results('Perovskite', analytical_results['perovskite_scatter_regions'])
        
        # Transferability Comparison
        if 'silicon_scatter_regions' in analytical_results and 'perovskite_scatter_regions' in analytical_results:
            html += self._create_transferability_comparison_table(analytical_results)
        
        html += "</div>"
        return html
    
    def _format_scatter_region_results(self, module_type: str, results: Dict[str, Any]) -> str:
        """Format scatter region analysis results for HTML"""
        html = f"""
        <h4>{module_type} Modules - Region Performance Analysis</h4>
        <table class="metrics-table">
            <thead>
                <tr>
                    <th>Power Region</th>
                    <th>Samples</th>
                    <th>MAE</th>
                    <th>RMSE</th>
                    <th>R²</th>
                    <th>Bias</th>
                    <th>Within 5%</th>
                    <th>Within 10%</th>
                    <th>Within 20%</th>
                </tr>
            </thead>
            <tbody>
        """
        
        regions = results.get('regions', {})
        for region_name, region_data in regions.items():
            region_display = region_name.replace('_', ' ').title()
            html += f"""
                    <tr>
                        <td>{region_display} ({region_data['power_range']})</td>
                        <td>{region_data['sample_count']:,}</td>
                        <td>{region_data['mae']:.4f}</td>
                        <td>{region_data['rmse']:.4f}</td>
                        <td>{region_data['r2']:.4f}</td>
                        <td>{region_data['bias']:.4f}</td>
                        <td>{region_data['pct_within_5pct']:.1f}%</td>
                        <td>{region_data['pct_within_10pct']:.1f}%</td>
                        <td>{region_data['pct_within_20pct']:.1f}%</td>
                    </tr>
            """
        
        # Overall results
        overall = results.get('overall', {})
        html += f"""
                    <tr class="overall-row">
                        <td><strong>Overall</strong></td>
                        <td><strong>{overall.get('total_samples', 0):,}</strong></td>
                        <td><strong>{overall.get('mae', 0):.4f}</strong></td>
                        <td><strong>{overall.get('rmse', 0):.4f}</strong></td>
                        <td><strong>{overall.get('r2', 0):.4f}</strong></td>
                        <td><strong>{overall.get('bias', 0):.4f}</strong></td>
                        <td><strong>{overall.get('pct_within_5pct', 0):.1f}%</strong></td>
                        <td><strong>{overall.get('pct_within_10pct', 0):.1f}%</strong></td>
                        <td><strong>{overall.get('pct_within_20pct', 0):.1f}%</strong></td>
                    </tr>
        """
        
        html += """
            </tbody>
        </table>
        """
        
        return html
    
    def _create_transferability_comparison_table(self, analytical_results: Dict[str, Any]) -> str:
        """Create comparison table for transferability analysis"""
        si_results = analytical_results.get('silicon_scatter_regions', {})
        pvk_results = analytical_results.get('perovskite_scatter_regions', {})
        
        html = """
        <h4>Transferability Analysis - Silicon vs Perovskite Performance Comparison</h4>
        <table class="metrics-table">
            <thead>
                <tr>
                    <th>Power Region</th>
                    <th colspan="2">MAE</th>
                    <th colspan="2">R²</th>
                    <th colspan="2">Bias</th>
                    <th colspan="2">Within 10%</th>
                </tr>
                <tr>
                    <th></th>
                    <th>Silicon</th>
                    <th>Perovskite</th>
                    <th>Silicon</th>
                    <th>Perovskite</th>
                    <th>Silicon</th>
                    <th>Perovskite</th>
                    <th>Silicon</th>
                    <th>Perovskite</th>
                </tr>
            </thead>
            <tbody>
        """
        
        si_regions = si_results.get('regions', {})
        pvk_regions = pvk_results.get('regions', {})
        
        for region_name in ['very_low', 'low', 'medium', 'high', 'very_high', 'extreme']:
            region_display = region_name.replace('_', ' ').title()
            
            si_data = si_regions.get(region_name, {})
            pvk_data = pvk_regions.get(region_name, {})
            
            html += f"""
                    <tr>
                        <td>{region_display}</td>
                        <td>{si_data.get('mae', 0):.4f}</td>
                        <td>{pvk_data.get('mae', 0):.4f}</td>
                        <td>{si_data.get('r2', 0):.4f}</td>
                        <td>{pvk_data.get('r2', 0):.4f}</td>
                        <td>{si_data.get('bias', 0):.4f}</td>
                        <td>{pvk_data.get('bias', 0):.4f}</td>
                        <td>{si_data.get('pct_within_10pct', 0):.1f}%</td>
                        <td>{pvk_data.get('pct_within_10pct', 0):.1f}%</td>
                    </tr>
            """
        
        # Overall comparison
        si_overall = si_results.get('overall', {})
        pvk_overall = pvk_results.get('overall', {})
        
        html += f"""
                    <tr class="overall-row">
                        <td><strong>Overall</strong></td>
                        <td><strong>{si_overall.get('mae', 0):.4f}</strong></td>
                        <td><strong>{pvk_overall.get('mae', 0):.4f}</strong></td>
                        <td><strong>{si_overall.get('r2', 0):.4f}</strong></td>
                        <td><strong>{pvk_overall.get('r2', 0):.4f}</strong></td>
                        <td><strong>{si_overall.get('bias', 0):.4f}</strong></td>
                        <td><strong>{pvk_overall.get('bias', 0):.4f}</strong></td>
                        <td><strong>{si_overall.get('pct_within_10pct', 0):.1f}%</strong></td>
                        <td><strong>{pvk_overall.get('pct_within_10pct', 0):.1f}%</strong></td>
                    </tr>
        """
        
        html += """
            </tbody>
        </table>
        """
        
        return html
    
    def _create_transferability_summary_section(self, analytical_results: Dict[str, Any]) -> str:
        """Create summary section for Si-Pvk model transferability"""
        html = """
        <div class="analytical-package transferability-summary">
            <h3>Model Transferability Summary: Silicon to Perovskite</h3>
            <p>Comparative analysis to evaluate how well the LSTM model transfers from Silicon to Perovskite modules across different analytical dimensions.</p>
        """
        
        # Create comparison table
        html += self._create_transferability_comparison_table(analytical_results)
        
        # Add key insights
        html += self._create_transferability_insights(analytical_results)
        
        html += "</div>"
        return html
    
    def _format_variability_results(self, module_type: str, results: Dict[str, Any]) -> str:
        """Format variability analysis results as HTML table"""
        if 'summary_stats' not in results:
            return f"<p>No variability data available for {module_type} modules.</p>"
        
        stats = results['summary_stats']
        
        html = f"""
        <h4>{module_type} Modules - Variability Analysis</h4>
        <table class="metrics-table">
            <tr><th>Metric</th><th>Value</th><th>Description</th></tr>
            <tr><td>Mean CV Error</td><td>{stats.get('mean_cv_error_relative', 0):.3f}</td><td>Average relative error in CV prediction</td></tr>
            <tr><td>CV Correlation</td><td>{stats.get('cv_correlation', 0):.3f}</td><td>Correlation between predicted and actual CV</td></tr>
            <tr><td>Mean Predicted CV</td><td>{stats.get('mean_predicted_cv', 0):.3f}</td><td>Average predicted coefficient of variation</td></tr>
            <tr><td>Mean Actual CV</td><td>{stats.get('mean_actual_cv', 0):.3f}</td><td>Average actual coefficient of variation</td></tr>
            <tr><td>Problematic Days</td><td>{len(results.get('problematic_days', []))}/{stats.get('n_days', 0)}</td><td>Days with high CV prediction errors</td></tr>
        </table>
        """
        
        return html
    
    def _format_ramp_rate_results(self, module_type: str, results: Dict[str, Any]) -> str:
        """Format ramp-rate analysis results as HTML table"""
        if 'summary_stats' not in results:
            return f"<p>No ramp-rate data available for {module_type} modules.</p>"
        
        stats = results['summary_stats']
        
        html = f"""
        <h4>{module_type} Modules - Ramp-Rate Analysis</h4>
        <table class="metrics-table">
            <tr><th>Metric</th><th>Value</th><th>Description</th></tr>
            <tr><td>Mean Ramp Error (Relative)</td><td>{stats.get('mean_ramp_error_mean_rel', 0):.3f}</td><td>Average relative error in ramp-rate prediction</td></tr>
            <tr><td>Ramp Correlation (Mean)</td><td>{stats.get('ramp_correlation_mean', 0):.3f}</td><td>Correlation for mean ramp-rate predictions</td></tr>
            <tr><td>Ramp Correlation (Std)</td><td>{stats.get('ramp_correlation_std', 0):.3f}</td><td>Correlation for ramp-rate variability predictions</td></tr>
            <tr><td>Mean Predicted Ramp</td><td>{stats.get('mean_predicted_ramp', 0):.3f}</td><td>Average predicted ramp-rate</td></tr>
            <tr><td>Mean Actual Ramp</td><td>{stats.get('mean_actual_ramp', 0):.3f}</td><td>Average actual ramp-rate</td></tr>
            <tr><td>Problematic Days</td><td>{len(results.get('problematic_days', []))}/{stats.get('n_days', 0)}</td><td>Days with high ramp-rate prediction errors</td></tr>
        </table>
        """
        
        return html
    
    def _format_intermittency_results(self, module_type: str, results: Dict[str, Any]) -> str:
        """Format intermittency analysis results as HTML table"""
        if not results or len(results) == 0:
            return f"<p>No intermittency analysis results available for {module_type} modules.</p>"
        
        # Debug information
        print(f"[DEBUG] Intermittency results for {module_type}: {list(results.keys())}")
        
        html = f"<h4>{module_type} Modules - Intermittency Analysis</h4>"
        
        # Check if we have prediction quality data
        if 'prediction_quality' in results:
            stats = results['prediction_quality']
            html += f"""
            <table class="metrics-table">
                <tr><th>Metric</th><th>Value</th><th>Description</th></tr>
                <tr><td>Mean Intermittency Error</td><td>{stats.get('mean_intermittency_error', 'N/A') if isinstance(stats.get('mean_intermittency_error', 'N/A'), str) else f"{stats.get('mean_intermittency_error', 0):.4f}"}</td><td>Average error in intermittency pattern prediction</td></tr>
                <tr><td>Transition Accuracy</td><td>{stats.get('transition_accuracy', 'N/A') if isinstance(stats.get('transition_accuracy', 'N/A'), str) else f"{stats.get('transition_accuracy', 0):.4f}"}</td><td>Accuracy of predicting power transitions</td></tr>
                <tr><td>Cloud Fluctuation Correlation</td><td>{stats.get('cloud_fluctuation_correlation', 'N/A') if isinstance(stats.get('cloud_fluctuation_correlation', 'N/A'), str) else f"{stats.get('cloud_fluctuation_correlation', 0):.4f}"}</td><td>Correlation for cloud-induced fluctuation prediction</td></tr>
                <tr><td>Problematic Periods</td><td>{len(results.get('problematic_periods', []))}</td><td>Periods with high intermittency prediction errors</td></tr>
            </table>
            """
            
            # Add intermittency metrics if available
            if 'intermittency_metrics' in results:
                metrics = results['intermittency_metrics']
                html += f"""
                <h5>Intermittency Metrics</h5>
                <table class="metrics-table">
                    <tr><th>Metric</th><th>Actual</th><th>Predicted</th><th>Error</th></tr>
                    <tr><td>Intermittency Ratio</td><td>{metrics.get('actual_intermittency_ratio', 'N/A') if isinstance(metrics.get('actual_intermittency_ratio', 'N/A'), str) else f"{metrics.get('actual_intermittency_ratio', 0):.4f}"}</td><td>{metrics.get('predicted_intermittency_ratio', 'N/A') if isinstance(metrics.get('predicted_intermittency_ratio', 'N/A'), str) else f"{metrics.get('predicted_intermittency_ratio', 0):.4f}"}</td><td>{abs(metrics.get('actual_intermittency_ratio', 0) - metrics.get('predicted_intermittency_ratio', 0)):.4f}</td></tr>
                    <tr><td>On Duration (avg)</td><td>{metrics.get('actual_on_duration', 'N/A') if isinstance(metrics.get('actual_on_duration', 'N/A'), str) else f"{metrics.get('actual_on_duration', 0):.2f}"}</td><td>{metrics.get('predicted_on_duration', 'N/A') if isinstance(metrics.get('predicted_on_duration', 'N/A'), str) else f"{metrics.get('predicted_on_duration', 0):.2f}"}</td><td>{abs(metrics.get('actual_on_duration', 0) - metrics.get('predicted_on_duration', 0)):.2f}</td></tr>
                    <tr><td>Off Duration (avg)</td><td>{metrics.get('actual_off_duration', 'N/A') if isinstance(metrics.get('actual_off_duration', 'N/A'), str) else f"{metrics.get('actual_off_duration', 0):.2f}"}</td><td>{metrics.get('predicted_off_duration', 'N/A') if isinstance(metrics.get('predicted_off_duration', 'N/A'), str) else f"{metrics.get('predicted_off_duration', 0):.2f}"}</td><td>{abs(metrics.get('actual_off_duration', 0) - metrics.get('predicted_off_duration', 0)):.2f}</td></tr>
                    <tr><td>Power Variability (on)</td><td>{metrics.get('actual_on_power_std', 'N/A') if isinstance(metrics.get('actual_on_power_std', 'N/A'), str) else f"{metrics.get('actual_on_power_std', 0):.4f}"}</td><td>{metrics.get('predicted_on_power_std', 'N/A') if isinstance(metrics.get('predicted_on_power_std', 'N/A'), str) else f"{metrics.get('predicted_on_power_std', 0):.4f}"}</td><td>{abs(metrics.get('actual_on_power_std', 0) - metrics.get('predicted_on_power_std', 0)):.4f}</td></tr>
            </table>
            """
        else:
            # Try to show other available data
            html += "<table class='metrics-table'><tr><th>Available Data</th><th>Status</th></tr>"
            for key, value in results.items():
                if key != 'module_type':
                    html += f"<tr><td>{key.replace('_', ' ').title()}</td><td>{'Available' if value is not None else 'Missing'}</td></tr>"
            html += "</table>"
        
        return html
    
    def _format_power_level_results(self, module_type: str, results: Dict[str, Any]) -> str:
        """Format power-level analysis results as HTML table"""
        if not results or len(results) == 0:
            return f"<p>No power-level analysis results available for {module_type} modules.</p>"
        
        # Debug information
        print(f"[DEBUG] Power-level results for {module_type}: {list(results.keys())}")
        
        html = f"<h4>{module_type} Modules - Power-Level Analysis</h4>"
        
        # Check if we have the expected data structure
        if 'level_analysis' in results and 'power_levels' in results:
            level_analysis = results['level_analysis']
            power_levels = results['power_levels']
            
            try:
                # Calculate summary statistics from the available data
                total_data_points = sum(level_mask.sum() for level_mask in power_levels.values() if level_mask is not None)
                n_power_levels = len(power_levels)
                
                html += f"""
                <table class="metrics-table">
                    <tr><th>Metric</th><th>Value</th><th>Description</th></tr>
                    <tr><td>Power Levels Analyzed</td><td>{n_power_levels}</td><td>Number of power level categories analyzed</td></tr>
                    <tr><td>Total Data Points</td><td>{total_data_points}</td><td>Total number of data points analyzed</td></tr>
                    <tr><th colspan='3'>Power Level Distribution</th></tr>
                """
                
                # Add power level distribution
                for level_name, level_mask in power_levels.items():
                    try:
                        count = level_mask.sum() if level_mask is not None else 0
                        percentage = count / total_data_points * 100 if total_data_points > 0 else 0
                        html += f"<tr><td>{level_name.replace('_', ' ').title()}</td><td>{count} points ({percentage:.1f}%)</td><td>Power generation level category</td></tr>"
                    except Exception as e:
                        html += f"<tr><td>{level_name.replace('_', ' ').title()}</td><td>Error: {str(e)}</td><td>Power generation level category</td></tr>"
                
                # Add level-specific metrics if available
                if level_analysis:
                    html += "<tr><th colspan='3'>Prediction Quality by Power Level</th></tr>"
                    for level_name, level_data in level_analysis.items():
                        try:
                            if isinstance(level_data, dict) and 'error_metrics' in level_data:
                                mae = level_data['error_metrics'].get('mae', 'N/A')
                                rmse = level_data['error_metrics'].get('rmse', 'N/A')
                                html += f"<tr><td>{level_name.replace('_', ' ').title()} MAE</td><td>{mae}</td><td>Mean Absolute Error for this power level</td></tr>"
                                html += f"<tr><td>{level_name.replace('_', ' ').title()} RMSE</td><td>{rmse}</td><td>Root Mean Squared Error for this power level</td></tr>"
                        except Exception as e:
                            html += f"<tr><td>{level_name.replace('_', ' ').title()}</td><td>Error: {str(e)}</td><td>Error in processing level data</td></tr>"
                
                html += "</table>"
                
            except Exception as e:
                html += f"<p>Error processing power-level data: {str(e)}</p>"
                
        else:
            # Show available data structure
            html += "<table class='metrics-table'><tr><th>Available Data</th><th>Status</th></tr>"
            for key, value in results.items():
                if key != 'module_type':
                    html += f"<tr><td>{key.replace('_', ' ').title()}</td><td>{'Available' if value is not None else 'Missing'}</td></tr>"
            html += "</table>"
        
        return html
    
    def _create_transferability_comparison_table(self, analytical_results: Dict[str, Any]) -> str:
        """Create comparison table for Si-Pvk transferability"""
        html = """
        <h4>Transferability Comparison Table</h4>
        <table class="metrics-table">
            <tr><th>Analysis Package</th><th>Silicon Performance</th><th>Perovskite Performance</th><th>Transferability Score</th><th>Key Insight</th></tr>
        """
        
        # Variability Analysis Comparison
        si_var = analytical_results.get('silicon_variability', {}).get('summary_stats', {})
        pvk_var = analytical_results.get('perovskite_variability', {}).get('summary_stats', {})
        
        if si_var and pvk_var:
            si_cv_corr = si_var.get('cv_correlation', 0)
            pvk_cv_corr = pvk_var.get('cv_correlation', 0)
            transferability_cv = min(si_cv_corr, pvk_cv_corr) / max(si_cv_corr, pvk_cv_corr) if max(si_cv_corr, pvk_cv_corr) > 0 else 0
            
            html += f"""
            <tr>
                <td>Variability Analysis</td>
                <td>CV Correlation: {si_cv_corr:.3f}</td>
                <td>CV Correlation: {pvk_cv_corr:.3f}</td>
                <td>{transferability_cv:.3f}</td>
                <td>{'Good' if transferability_cv > 0.8 else 'Moderate' if transferability_cv > 0.6 else 'Poor'} transferability</td>
            </tr>
            """
        
        # Ramp-Rate Analysis Comparison
        si_ramp = analytical_results.get('silicon_ramp_rate', {}).get('summary_stats', {})
        pvk_ramp = analytical_results.get('perovskite_ramp_rate', {}).get('summary_stats', {})
        
        if si_ramp and pvk_ramp:
            si_ramp_corr = si_ramp.get('ramp_correlation_mean', 0)
            pvk_ramp_corr = pvk_ramp.get('ramp_correlation_mean', 0)
            transferability_ramp = min(si_ramp_corr, pvk_ramp_corr) / max(si_ramp_corr, pvk_ramp_corr) if max(si_ramp_corr, pvk_ramp_corr) > 0 else 0
            
            html += f"""
            <tr>
                <td>Ramp-Rate Analysis</td>
                <td>Ramp Correlation: {si_ramp_corr:.3f}</td>
                <td>Ramp Correlation: {pvk_ramp_corr:.3f}</td>
                <td>{transferability_ramp:.3f}</td>
                <td>{'Good' if transferability_ramp > 0.8 else 'Moderate' if transferability_ramp > 0.6 else 'Poor'} transferability</td>
            </tr>
            """
        
        # Intermittency Analysis Comparison
        si_inter = analytical_results.get('silicon_intermittency', {}).get('summary_stats', {})
        pvk_inter = analytical_results.get('perovskite_intermittency', {}).get('summary_stats', {})
        
        if si_inter and pvk_inter:
            si_inter_err = si_inter.get('mean_intermittency_error', 0)
            pvk_inter_err = pvk_inter.get('mean_intermittency_error', 0)
            # Lower error is better, so we calculate transferability differently
            avg_error = (si_inter_err + pvk_inter_err) / 2
            transferability_inter = 1 - abs(si_inter_err - pvk_inter_err) if avg_error > 0 else 0
            
            html += f"""
            <tr>
                <td>Intermittency Analysis</td>
                <td>Intermittency Error: {si_inter_err:.3f}</td>
                <td>Intermittency Error: {pvk_inter_err:.3f}</td>
                <td>{transferability_inter:.3f}</td>
                <td>{'Good' if transferability_inter > 0.8 else 'Moderate' if transferability_inter > 0.6 else 'Poor'} transferability</td>
            </tr>
            """
        
        # Power-Level Analysis Comparison
        si_power = analytical_results.get('silicon_power_level', {})
        pvk_power = analytical_results.get('perovskite_power_level', {})
        
        if si_power and pvk_power:
            # Calculate from the actual data structure
            si_levels = si_power.get('power_levels', {})
            pvk_levels = pvk_power.get('power_levels', {})
            
            si_total = sum(level_mask.sum() for level_mask in si_levels.values()) if si_levels else 0
            pvk_total = sum(level_mask.sum() for level_mask in pvk_levels.values()) if pvk_levels else 0
            si_n_levels = len(si_levels)
            pvk_n_levels = len(pvk_levels)
            
            # Compare data coverage
            coverage_similarity = min(si_total, pvk_total) / max(si_total, pvk_total) if max(si_total, pvk_total) > 0 else 0
            level_similarity = min(si_n_levels, pvk_n_levels) / max(si_n_levels, pvk_n_levels) if max(si_n_levels, pvk_n_levels) > 0 else 0
            transferability_power = (coverage_similarity + level_similarity) / 2
            
            html += f"""
            <tr>
                <td>Power-Level Analysis</td>
                <td>Data Points: {si_total}, Levels: {si_n_levels}</td>
                <td>Data Points: {pvk_total}, Levels: {pvk_n_levels}</td>
                <td>{transferability_power:.3f}</td>
                <td>{'Good' if transferability_power > 0.8 else 'Moderate' if transferability_power > 0.6 else 'Poor'} data similarity</td>
            </tr>
            """
        
        html += "</table>"
        return html
    
    def _create_transferability_insights(self, analytical_results: Dict[str, Any]) -> str:
        """Create key insights section for transferability"""
        html = """
        <h4>Key Transferability Insights</h4>
        <div class="insights-box">
            <ul>
        """
        
        # Generate insights based on analytical results
        insights = []
        
        # CV Analysis Insight
        si_var = analytical_results.get('silicon_variability', {}).get('summary_stats', {})
        pvk_var = analytical_results.get('perovskite_variability', {}).get('summary_stats', {})
        
        if si_var and pvk_var:
            si_cv_corr = si_var.get('cv_correlation', 0)
            pvk_cv_corr = pvk_var.get('cv_correlation', 0)
            if abs(si_cv_corr - pvk_cv_corr) < 0.1:
                insights.append("Variability prediction shows strong transferability between module types")
            else:
                insights.append(f"Variability prediction differs significantly between module types (Δ={abs(si_cv_corr - pvk_cv_corr):.3f})")
        
        # Ramp-Rate Analysis Insight
        si_ramp = analytical_results.get('silicon_ramp_rate', {}).get('summary_stats', {})
        pvk_ramp = analytical_results.get('perovskite_ramp_rate', {}).get('summary_stats', {})
        
        if si_ramp and pvk_ramp:
            si_ramp_err = si_ramp.get('mean_ramp_error_mean_rel', 0)
            pvk_ramp_err = pvk_ramp.get('mean_ramp_error_mean_rel', 0)
            if abs(si_ramp_err - pvk_ramp_err) < 0.05:
                insights.append("Ramp-rate prediction quality is consistent across module types")
            else:
                insights.append(f"Ramp-rate prediction shows module-specific challenges (Δ={abs(si_ramp_err - pvk_ramp_err):.3f})")
        
        # Intermittency Analysis Insight
        si_inter = analytical_results.get('silicon_intermittency', {}).get('summary_stats', {})
        pvk_inter = analytical_results.get('perovskite_intermittency', {}).get('summary_stats', {})
        
        if si_inter and pvk_inter:
            si_inter_err = si_inter.get('mean_intermittency_error', 0)
            pvk_inter_err = pvk_inter.get('mean_intermittency_error', 0)
            if abs(si_inter_err - pvk_inter_err) < 0.02:
                insights.append("Intermittency prediction shows similar performance across module types")
            else:
                insights.append(f"Intermittency prediction varies between module types (Δ={abs(si_inter_err - pvk_inter_err):.3f})")
        
        # Power-Level Analysis Insight
        si_power = analytical_results.get('silicon_power_level', {})
        pvk_power = analytical_results.get('perovskite_power_level', {})
        
        if si_power and pvk_power:
            si_levels = si_power.get('power_levels', {})
            pvk_levels = pvk_power.get('power_levels', {})
            
            si_total = sum(level_mask.sum() for level_mask in si_levels.values()) if si_levels else 0
            pvk_total = sum(level_mask.sum() for level_mask in pvk_levels.values()) if pvk_levels else 0
            ratio = min(si_total, pvk_total) / max(si_total, pvk_total) if max(si_total, pvk_total) > 0 else 0
            
            if ratio > 0.8:
                insights.append("Similar data coverage across module types enables good transferability assessment")
            else:
                insights.append(f"Uneven data coverage may limit transferability assessment (ratio: {ratio:.2f})")
        
        # Add overall assessment
        if len(insights) >= 2:
            insights.append("The analytical packages provide comprehensive insights into model transferability from Silicon to Perovskite modules")
        
        # Add insights to HTML
        for insight in insights:
            html += f"<li>{insight}</li>"
        
        html += """
            </ul>
        </div>
        """
        
        return html
    
    def save_pdf_plots(self, evaluation_results=None) -> str:
        """
        Generate PDF plots for forecast evaluation results
        
        Args:
            evaluation_results: Dictionary containing evaluation results for both module types
            
        Returns:
            Path to the generated PDF file
        """
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        report_dir = self.output_dir / self.model_name / timestamp
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate scatter plots for both module types
        self._create_scatter_plots_pdf(evaluation_results, report_dir)
        
        # Generate time series comparison plots
        self._create_time_series_plots_pdf(evaluation_results, report_dir)
        
        # Generate error distribution plots
        self._create_error_distribution_plots_pdf(evaluation_results, report_dir)
        
        # Generate comparison difference plots (2 subplots)
        self._create_comparison_difference_plots_pdf(evaluation_results, report_dir)
        
        print(f"[INFO] PDF plots saved to: {report_dir}")
        return str(report_dir)
    
    def save_analytical_pdf_plots(self, analytical_results=None) -> str:
        """
        Generate PDF plots for analytical packages
        
        Args:
            analytical_results: Dictionary containing analytical results for both module types
            
        Returns:
            Path to the generated PDF files
        """
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        report_dir = self.output_dir / self.model_name / timestamp
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate analytical package difference plots
        self._create_analytical_package_plots_pdf(analytical_results, report_dir)
        
        print(f"[INFO] Analytical package PDF plots saved to: {report_dir}")
        return str(report_dir)
    
    def _create_scatter_plots_pdf(self, evaluation_results, report_dir):
        """Create scatter plots in PDF format similar to experimental scenarios"""
        import matplotlib.pyplot as plt
        
        if not evaluation_results:
            print("[WARN] No evaluation results available for scatter plot generation")
            return
            
        for module_type, module_data in evaluation_results.items():
            if not module_data or 'predictions' not in module_data or 'actuals' not in module_data:
                continue
                
            predictions = module_data['predictions']
            actuals = module_data['actuals']
            
            # Flatten predictions and actuals for scatter plot 
            if len(predictions.shape) > 2:
                predictions_flat = predictions.reshape(-1, predictions.shape[-1])
                actuals_flat = actuals.reshape(-1, actuals.shape[-1])
            else:
                predictions_flat = predictions
                actuals_flat = actuals
            
            # Create scatter plot
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            
            # Scatter plot
            ax.scatter(actuals_flat[:, 0], predictions_flat[:, 0], 
                      alpha=0.6, s=20, color='blue', edgecolors='darkblue', linewidth=0.5)
            
            # Perfect prediction line
            min_val = min(actuals_flat[:, 0].min(), predictions_flat[:, 0].min())
            max_val = max(actuals_flat[:, 0].max(), predictions_flat[:, 0].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=4, label='Perfect Prediction')
            
            # Formatting - larger fonts, no statistics box
            ax.set_title(f'{module_type.upper()} Modules - Predicted vs Actual', 
                        fontsize=28, fontweight='bold', pad=20)
            ax.set_xlabel('Actual Power Output (normal.)', fontsize=24, fontweight='bold')
            ax.set_ylabel('Predicted Power Output (normal.)', fontsize=24, fontweight='bold')
            ax.legend(fontsize=20)
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            ax.tick_params(axis='both', which='major', labelsize=20)
            
            plt.tight_layout()
            
            # Save as PDF
            scatter_pdf = report_dir / f"scatter_plot_{module_type}.pdf"
            plt.savefig(scatter_pdf, format='pdf', bbox_inches='tight', facecolor='white')
            print(f"[INFO] Scatter plot (PDF) saved to: {scatter_pdf}")
            
            plt.close(fig)
    
    def _create_time_series_plots_pdf(self, evaluation_results, report_dir, start_date=None, end_date=None):
        """Create time series comparison plots for specified period"""
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from datetime import datetime
        
        if not evaluation_results:
            print("[WARN] No evaluation results available for time series plot generation")
            return
            
        # Default analysis period (August 19-24, 2025) if not specified
        if start_date is None:
            start_date = datetime(2025, 8, 19)
        if end_date is None:
            end_date = datetime(2025, 8, 24)
        
        for module_type, module_data in evaluation_results.items():
            if not module_data or 'predictions' not in module_data or 'actuals' not in module_data:
                continue
                
            # Get predictions, actuals, and timestamps from evaluation results (already aligned)
            predictions = module_data['predictions']
            actuals = module_data['actuals']
            timestamps = module_data.get('timestamps', None)
            
            # Flatten if needed
            if len(predictions.shape) > 2:
                predictions_flat = predictions.reshape(-1, predictions.shape[-1])
                actuals_flat = actuals.reshape(-1, actuals.shape[-1])
            else:
                predictions_flat = predictions
                actuals_flat = actuals
            
            # Use timestamps from evaluation results if available
            if timestamps is not None:
                # Convert to pandas datetime if needed
                if not isinstance(timestamps, pd.DatetimeIndex):
                    timestamps = pd.to_datetime(timestamps)
                
                # Filter for the analysis period
                period_mask = (timestamps >= start_date) & (timestamps <= end_date)
                period_indices = np.where(period_mask)[0]
                
                if len(period_indices) > 0:
                    actuals_period = actuals_flat[period_indices, 0]
                    predictions_period = predictions_flat[period_indices, 0]
                    timestamps_period = timestamps[period_indices]
                else:
                    print(f"[WARN] No data found for {module_type} in the specified period")
                    continue
            else:
                print(f"[WARN] No timestamps available for {module_type}")
                continue
            
            # Load weather data for the specific period
            try:
                feats = []
                if self.preprocessor is not None:
                    feats = list(self.preprocessor.feature_scalers.keys()) + list(self.preprocessor.output_scalers.keys())
                
                # Load data for the specific module type
                if module_type == "silicon":
                    test_df = self.load_weather_integrated_data(feats, module_type="silicon")
                else:
                    test_df = self.load_weather_integrated_data(feats, module_type="perovskite")
                
                test_df = test_df.sort_values('timestamp')
                test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
                
                # Filter weather data for the same period
                weather_mask = (test_df['timestamp'] >= start_date) & (test_df['timestamp'] <= end_date)
                period_df = test_df[weather_mask].copy()
                
                if len(period_df) == 0:
                    print(f"[WARN] No weather data found for {module_type} in the specified period")
                    continue
                    
            except Exception as e:
                print(f"[WARN] Could not load weather data for {module_type}: {e}")
                continue
            
            # Create subplots
            fig, axes = plt.subplots(3, 1, figsize=(15, 12))
            
            # Panel 1: P_normalized - Time Series Comparison
            ax1 = axes[0]
            
            # Plot actual and predicted power using actual timestamps
            ax1.plot(timestamps_period, actuals_period, 'b-', label=f'{module_type.upper()} P_normalized Actual', 
                    linewidth=2, alpha=0.8)
            ax1.plot(timestamps_period, predictions_period, 'r--', label=f'{module_type.upper()} P_normalized Predicted', 
                    linewidth=2, alpha=0.8)
            
            ax1.set_title(f'P_normalized - Time Series Comparison - {module_type.upper()}', 
                         fontsize=20, fontweight='bold')
            ax1.set_ylabel('Normalized Power', fontsize=18, fontweight='bold')
            ax1.legend(fontsize=16)
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='both', which='major', labelsize=16)
            
            # Panel 2: P_normalized - Difference Analysis
            ax2 = axes[1]
            
            # Calculate differences and errors
            prediction_error = predictions_period - actuals_period
            
            ax2.plot(timestamps_period, prediction_error, 'r-', label=f'P_normalized Prediction Error ({module_type.upper()})', 
                    linewidth=2, alpha=0.8)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            
            ax2.set_title(f'P_normalized - Difference Analysis - {module_type.upper()}', 
                         fontsize=20, fontweight='bold')
            ax2.set_ylabel('Difference', fontsize=18, fontweight='bold')
            ax2.legend(fontsize=16)
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis='both', which='major', labelsize=16)
            
            # Panel 3: Weather Features
            ax3 = axes[2]
            
            # Plot weather features
            weather_features = ['TT_10', 'RF_10', 'RWS_10', 'RWS_IND_10', 'V_N']
            colors = ['purple', 'lightgreen', 'lightblue', 'cyan', 'brown']
            
            for feature, color in zip(weather_features, colors):
                if feature in period_df.columns:
                    ax3.plot(period_df['timestamp'].values, period_df[feature].values, 
                            color=color, label=feature, linewidth=2, alpha=0.8)
            
            # Add irradiance on secondary y-axis
            if 'Irr' in period_df.columns:
                ax3_secondary = ax3.twinx()
                irr_values = period_df['Irr'].values
                ax3_secondary.plot(period_df['timestamp'].values, irr_values, 
                                 'orange', label='Irr', linewidth=2, alpha=0.8)
                ax3_secondary.set_ylim(bottom=0, top=1200)  # Extend from 100 to 120 for legend space
                ax3_secondary.set_ylabel('Irradiance (W/m²)', fontsize=18, fontweight='bold')
                ax3_secondary.tick_params(axis='y', which='major', labelsize=16)
                
                # Combine legends and position in upper extended area
                lines1, labels1 = ax3.get_legend_handles_labels()
                lines2, labels2 = ax3_secondary.get_legend_handles_labels()
                ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=13, 
                          loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=6)
            else:
                ax3.legend(fontsize=13, loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=6)
            
            # Extend Y-axis upward to create space for legend
            ax3.set_ylim(bottom=0, top=120)  # Extend from 100 to 120 for legend space
            
            ax3.set_title('Weather Features', fontsize=20, fontweight='bold')
            ax3.set_xlabel('Time', fontsize=18, fontweight='bold')
            ax3.set_ylabel('Weather Values', fontsize=18, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            ax3.tick_params(axis='both', which='major', labelsize=16)
            
            # Format x-axis for timestamps
            ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            ax3.xaxis.set_major_locator(mdates.HourLocator(interval=6))
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            
            # Save as PDF
            timeseries_pdf = report_dir / f"time_series_analysis_{module_type}.pdf"
            plt.savefig(timeseries_pdf, format='pdf', bbox_inches='tight', facecolor='white')
            print(f"[INFO] Time series analysis (PDF) saved to: {timeseries_pdf}")
            
            plt.close(fig)
    
    def _create_comparison_difference_plots_pdf(self, evaluation_results: Dict[str, Any], report_dir: Path):
        """Create comparison difference plots (2 subplots) for Si vs Pvk as PDF"""
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        
        # Define the analysis period (same as time series comparison)
        start_date = pd.Timestamp('2025-08-19')
        end_date = pd.Timestamp('2025-08-24')
        
        # Get data for both module types
        si_data = evaluation_results.get('silicon', {})
        pvk_data = evaluation_results.get('perovskite', {})
        
        if not si_data or not pvk_data:
            print("[WARN] Missing data for comparison difference plots")
            return
        
        # Get predictions, actuals, and timestamps
        si_predictions = si_data.get('predictions')
        si_actuals = si_data.get('actuals')
        si_timestamps = si_data.get('timestamps')
        
        pvk_predictions = pvk_data.get('predictions')
        pvk_actuals = pvk_data.get('actuals')
        pvk_timestamps = pvk_data.get('timestamps')
        
        if si_predictions is None or pvk_predictions is None:
            print("[WARN] Missing predictions for comparison difference plots")
            return
        
        # Flatten if needed
        if len(si_predictions.shape) > 2:
            si_predictions_flat = si_predictions.reshape(-1, si_predictions.shape[-1])
            si_actuals_flat = si_actuals.reshape(-1, si_actuals.shape[-1])
        else:
            si_predictions_flat = si_predictions
            si_actuals_flat = si_actuals
        
        if len(pvk_predictions.shape) > 2:
            pvk_predictions_flat = pvk_predictions.reshape(-1, pvk_predictions.shape[-1])
            pvk_actuals_flat = pvk_actuals.reshape(-1, pvk_actuals.shape[-1])
        else:
            pvk_predictions_flat = pvk_predictions
            pvk_actuals_flat = pvk_actuals
        
        # Convert timestamps to pandas datetime
        if si_timestamps is not None and not isinstance(si_timestamps, pd.DatetimeIndex):
            si_timestamps = pd.to_datetime(si_timestamps)
        if pvk_timestamps is not None and not isinstance(pvk_timestamps, pd.DatetimeIndex):
            pvk_timestamps = pd.to_datetime(pvk_timestamps)
        
        # Filter for the analysis period
        si_period_mask = (si_timestamps >= start_date) & (si_timestamps <= end_date)
        pvk_period_mask = (pvk_timestamps >= start_date) & (pvk_timestamps <= end_date)
        
        si_period_indices = np.where(si_period_mask)[0]
        pvk_period_indices = np.where(pvk_period_mask)[0]
        
        if len(si_period_indices) == 0 or len(pvk_period_indices) == 0:
            print("[WARN] No data found in the specified period for comparison plots")
            return
        
        # Get period data
        si_actuals_period = si_actuals_flat[si_period_indices, 0]
        si_predictions_period = si_predictions_flat[si_period_indices, 0]
        si_timestamps_period = si_timestamps[si_period_indices]
        
        pvk_actuals_period = pvk_actuals_flat[pvk_period_indices, 0]
        pvk_predictions_period = pvk_predictions_flat[pvk_period_indices, 0]
        pvk_timestamps_period = pvk_timestamps[pvk_period_indices]
        
        # Ensure both datasets have the same length (use minimum)
        min_length = min(len(si_actuals_period), len(pvk_actuals_period))
        si_actuals_period = si_actuals_period[:min_length]
        si_predictions_period = si_predictions_period[:min_length]
        si_timestamps_period = si_timestamps_period[:min_length]
        
        pvk_actuals_period = pvk_actuals_period[:min_length]
        pvk_predictions_period = pvk_predictions_period[:min_length]
        pvk_timestamps_period = pvk_timestamps_period[:min_length]
        
        # Calculate differences and errors
        actual_diff_si_pvk = si_actuals_period - pvk_actuals_period
        pred_error_si = si_predictions_period - si_actuals_period
        pred_error_pvk = pvk_predictions_period - pvk_actuals_period
        pred_error_diff_si_pvk = pred_error_si - pred_error_pvk
        
        # Load weather data for the period
        try:
            feats = []
            if self.preprocessor is not None:
                feats = list(self.preprocessor.feature_scalers.keys()) + list(self.preprocessor.output_scalers.keys())
            
            test_df = self.load_weather_integrated_data(feats, module_type="silicon")
            test_df = test_df.sort_values('timestamp')
            test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
            
            # Filter weather data for the same period
            weather_mask = (test_df['timestamp'] >= start_date) & (test_df['timestamp'] <= end_date)
            period_df = test_df[weather_mask].copy()
            
        except Exception as e:
            print(f"[WARN] Could not load weather data for comparison plots: {e}")
            return
        
        # Create 2 subplots
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # Panel 1: Actual Diff and Prediction Error Diff
        ax1 = axes[0]
        
        ax1.plot(si_timestamps_period, actual_diff_si_pvk, 'purple', 
                label='P_normalized Actual Diff (Si - Pvk)', linewidth=2, alpha=0.8)
        ax1.plot(si_timestamps_period, pred_error_diff_si_pvk, 'green', 
                label='P_normalized Prediction Error Diff (Si - Pvk)', linewidth=2, alpha=0.8)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        ax1.set_title('Difference Analysis - Silicon vs Perovskite', 
                     fontsize=20, fontweight='bold', pad=15)
        ax1.set_ylabel('Difference', fontsize=18, fontweight='bold')
        ax1.legend(fontsize=16, loc='lower right')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='both', which='major', labelsize=16)
        
        # Panel 2: Weather Features
        ax2 = axes[1]
        
        # Plot weather features
        weather_features = ['TT_10', 'RF_10', 'RWS_10', 'RWS_IND_10', 'V_N']
        colors = ['purple', 'lightgreen', 'lightblue', 'cyan', 'brown']
        
        # Extend Y-axis upward to create space for legend
        ax2.set_ylim(bottom=0, top=120)
        
        for feature, color in zip(weather_features, colors):
            if feature in period_df.columns:
                ax2.plot(period_df['timestamp'].values, period_df[feature].values, 
                        color=color, label=feature, linewidth=2, alpha=0.8)
        
        # Add irradiance on secondary y-axis
        if 'Irr' in period_df.columns:
            ax2_secondary = ax2.twinx()
            irr_values = period_df['Irr'].values
            ax2_secondary.plot(period_df['timestamp'].values, irr_values, 
                             'orange', label='Irr', linewidth=2, alpha=0.8)
            ax2_secondary.set_ylim(bottom=0, top=1200)
            ax2_secondary.set_ylabel('Irradiance (W/m²)', fontsize=18, fontweight='bold')
            ax2_secondary.tick_params(axis='y', which='major', labelsize=16)
            
            # Combine legends and position in upper extended area
            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = ax2_secondary.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=13, 
                      loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=6)
        else:
            ax2.legend(fontsize=13, loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=6)
        
        ax2.set_title('Weather Features', fontsize=20, fontweight='bold', pad=15)
        ax2.set_xlabel('Time', fontsize=18, fontweight='bold')
        ax2.set_ylabel('Weather Values', fontsize=18, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='both', which='major', labelsize=16)
        
        # Format x-axis for timestamps
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax2.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # Save as PDF
        comparison_pdf = report_dir / "comparison_difference_analysis.pdf"
        plt.savefig(comparison_pdf, format='pdf', bbox_inches='tight', facecolor='white')
        print(f"[INFO] Comparison difference analysis (PDF) saved to: {comparison_pdf}")
        
        plt.close(fig)
    
    def _create_analytical_package_plots_pdf(self, analytical_results: Dict[str, Any], report_dir: Path):
        """Create PDF plots for each analytical package showing Si vs Pvk differences"""
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        
        if not analytical_results:
            print("[WARN] No analytical results available for PDF plots")
            return
        
        # Package 1: Variability Analysis
        self._create_variability_diff_plot_pdf(analytical_results, report_dir)
        
        # Package 2: Ramp Rate Analysis
        self._create_ramp_rate_diff_plot_pdf(analytical_results, report_dir)
        
        # Package 3: Intermittency Analysis
        self._create_intermittency_diff_plot_pdf(analytical_results, report_dir)
        
        # Combined 4-panel plot with all packages + weather
        self._create_combined_analytical_plot_pdf(analytical_results, report_dir)
    
    def _create_variability_diff_plot_pdf(self, analytical_results: Dict[str, Any], report_dir: Path):
        """Create CV difference plot for Si vs Pvk"""
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        
        # Get data for both module types
        si_data = analytical_results.get('silicon_variability', {})
        pvk_data = analytical_results.get('perovskite_variability', {})
        
        if not si_data or not pvk_data:
            print("[WARN] Missing variability data for PDF plot")
            return
        
        si_daily = si_data.get('daily_data')
        pvk_daily = pvk_data.get('daily_data')
        
        if si_daily is None or pvk_daily is None or len(si_daily) == 0 or len(pvk_daily) == 0:
            print("[WARN] Missing daily CV data for PDF plot")
            return
        
        # Ensure both have the same dates (use intersection)
        si_dates = set(si_daily['date'].values)
        pvk_dates = set(pvk_daily['date'].values)
        common_dates = sorted(list(si_dates & pvk_dates))
        
        if len(common_dates) == 0:
            print("[WARN] No common dates for CV comparison")
            return
        
        # Filter to common dates
        si_daily_filtered = si_daily[si_daily['date'].isin(common_dates)].sort_values('date')
        pvk_daily_filtered = pvk_daily[pvk_daily['date'].isin(common_dates)].sort_values('date')
        
        # Calculate differences
        cv_actual_diff = si_daily_filtered['actual_cv'].values - pvk_daily_filtered['actual_cv'].values
        # Use pre-calculated error values
        cv_pred_error_diff = si_daily_filtered['cv_error_absolute'].values - pvk_daily_filtered['cv_error_absolute'].values
        
        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(15, 6))
        
        dates = pd.to_datetime(si_daily_filtered['date'].values)
        
        ax.plot(dates, cv_actual_diff, 'purple', 
                label='Daily CV Actual Diff (Si - Pvk)', linewidth=2, alpha=0.8)
        ax.plot(dates, cv_pred_error_diff, 'green', 
                label='Daily CV Prediction Error Diff (Si - Pvk)', linewidth=2, alpha=0.8)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        ax.set_title('Package 1: Variability Analysis - CV Difference (Si vs Pvk)', 
                     fontsize=20, fontweight='bold', pad=15)
        ax.set_xlabel('Date', fontsize=18, fontweight='bold')
        ax.set_ylabel('CV Difference', fontsize=18, fontweight='bold')
        ax.legend(fontsize=16, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=16)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # Save as PDF
        cv_pdf = report_dir / "analytical_package1_cv_difference.pdf"
        plt.savefig(cv_pdf, format='pdf', bbox_inches='tight', facecolor='white')
        print(f"[INFO] Package 1 CV difference plot (PDF) saved to: {cv_pdf}")
        
        plt.close(fig)
    
    def _create_ramp_rate_diff_plot_pdf(self, analytical_results: Dict[str, Any], report_dir: Path):
        """Create Ramp Rate difference plot for Si vs Pvk"""
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        
        # Get data for both module types
        si_data = analytical_results.get('silicon_ramp_rate', {})
        pvk_data = analytical_results.get('perovskite_ramp_rate', {})
        
        if not si_data or not pvk_data:
            print("[WARN] Missing ramp rate data for PDF plot")
            return
        
        si_daily = si_data.get('daily_data')
        pvk_daily = pvk_data.get('daily_data')
        
        if si_daily is None or pvk_daily is None or len(si_daily) == 0 or len(pvk_daily) == 0:
            print("[WARN] Missing daily ramp data for PDF plot")
            return
        
        # Ensure both have the same dates
        si_dates = set(si_daily['date'].values)
        pvk_dates = set(pvk_daily['date'].values)
        common_dates = sorted(list(si_dates & pvk_dates))
        
        if len(common_dates) == 0:
            print("[WARN] No common dates for ramp rate comparison")
            return
        
        # Filter to common dates
        si_daily_filtered = si_daily[si_daily['date'].isin(common_dates)].sort_values('date')
        pvk_daily_filtered = pvk_daily[pvk_daily['date'].isin(common_dates)].sort_values('date')
        
        # Calculate differences (using mean ramp rates)
        ramp_actual_diff = si_daily_filtered['actual_mean_ramp'].values - pvk_daily_filtered['actual_mean_ramp'].values
        # Use pre-calculated error values
        ramp_pred_error_diff = si_daily_filtered['ramp_error_mean_abs'].values - pvk_daily_filtered['ramp_error_mean_abs'].values
        
        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(15, 6))
        
        dates = pd.to_datetime(si_daily_filtered['date'].values)
        
        ax.plot(dates, ramp_actual_diff, 'purple', 
                label='Daily Ramp Actual Diff (Si - Pvk)', linewidth=2, alpha=0.8)
        ax.plot(dates, ramp_pred_error_diff, 'green', 
                label='Daily Ramp Prediction Error Diff (Si - Pvk)', linewidth=2, alpha=0.8)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        ax.set_title('Package 2: Ramp-Rate Analysis - Ramp Rate Difference (Si vs Pvk)', 
                     fontsize=20, fontweight='bold', pad=15)
        ax.set_xlabel('Date', fontsize=18, fontweight='bold')
        ax.set_ylabel('Ramp Rate Difference', fontsize=18, fontweight='bold')
        ax.legend(fontsize=16, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=16)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # Save as PDF
        ramp_pdf = report_dir / "analytical_package2_ramp_rate_difference.pdf"
        plt.savefig(ramp_pdf, format='pdf', bbox_inches='tight', facecolor='white')
        print(f"[INFO] Package 2 Ramp Rate difference plot (PDF) saved to: {ramp_pdf}")
        
        plt.close(fig)
    
    def _create_intermittency_diff_plot_pdf(self, analytical_results: Dict[str, Any], report_dir: Path):
        """Create Intermittency difference plot for Si vs Pvk"""
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        
        # Get data for both module types
        si_data = analytical_results.get('silicon_intermittency', {})
        pvk_data = analytical_results.get('perovskite_intermittency', {})
        
        if not si_data or not pvk_data:
            print("[WARN] Missing intermittency data for PDF plot")
            return
        
        si_daily = si_data.get('daily_intermittency')
        pvk_daily = pvk_data.get('daily_intermittency')
        
        if si_daily is None or pvk_daily is None or len(si_daily) == 0 or len(pvk_daily) == 0:
            print("[WARN] Missing daily intermittency data for PDF plot")
            return
        
        # Ensure both have the same dates
        si_dates = set(si_daily['date'].values)
        pvk_dates = set(pvk_daily['date'].values)
        common_dates = sorted(list(si_dates & pvk_dates))
        
        if len(common_dates) == 0:
            print("[WARN] No common dates for intermittency comparison")
            return
        
        # Filter to common dates
        si_daily_filtered = si_daily[si_daily['date'].isin(common_dates)].sort_values('date')
        pvk_daily_filtered = pvk_daily[pvk_daily['date'].isin(common_dates)].sort_values('date')
        
        # Calculate differences (using intermittency ratio)
        intermittency_actual_diff = si_daily_filtered['actual_intermittency_ratio'].values - pvk_daily_filtered['actual_intermittency_ratio'].values
        # Use pre-calculated error values
        intermittency_pred_error_diff = si_daily_filtered['intermittency_ratio_error'].values - pvk_daily_filtered['intermittency_ratio_error'].values
        
        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(15, 6))
        
        dates = pd.to_datetime(si_daily_filtered['date'].values)
        
        ax.plot(dates, intermittency_actual_diff, 'purple', 
                label='Daily Intermittency Actual Diff (Si - Pvk)', linewidth=2, alpha=0.8)
        ax.plot(dates, intermittency_pred_error_diff, 'green', 
                label='Daily Intermittency Prediction Error Diff (Si - Pvk)', linewidth=2, alpha=0.8)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        ax.set_title('Package 3: Intermittency Analysis - Intermittency Difference (Si vs Pvk)', 
                     fontsize=20, fontweight='bold', pad=15)
        ax.set_xlabel('Date', fontsize=18, fontweight='bold')
        ax.set_ylabel('Intermittency Ratio Difference', fontsize=18, fontweight='bold')
        ax.legend(fontsize=16, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=16)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # Save as PDF
        intermittency_pdf = report_dir / "analytical_package3_intermittency_difference.pdf"
        plt.savefig(intermittency_pdf, format='pdf', bbox_inches='tight', facecolor='white')
        print(f"[INFO] Package 3 Intermittency difference plot (PDF) saved to: {intermittency_pdf}")
        
        plt.close(fig)
    
    def _create_combined_analytical_plot_pdf(self, analytical_results: Dict[str, Any], report_dir: Path):
        """Create combined 4-panel plot with all analytical packages + weather features"""
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        
        print("[INFO] Creating combined analytical packages plot (PDF)...")
        
        # Get data for all three packages
        si_cv_data = analytical_results.get('silicon_variability', {}).get('daily_data')
        pvk_cv_data = analytical_results.get('perovskite_variability', {}).get('daily_data')
        
        si_ramp_data = analytical_results.get('silicon_ramp_rate', {}).get('daily_data')
        pvk_ramp_data = analytical_results.get('perovskite_ramp_rate', {}).get('daily_data')
        
        si_int_data = analytical_results.get('silicon_intermittency', {}).get('daily_intermittency')
        pvk_int_data = analytical_results.get('perovskite_intermittency', {}).get('daily_intermittency')
        
        # Check if all data is available
        if not all([si_cv_data is not None, pvk_cv_data is not None,
                    si_ramp_data is not None, pvk_ramp_data is not None,
                    si_int_data is not None, pvk_int_data is not None]):
            print("[WARN] Missing data for combined analytical plot")
            return
        
        # Find common dates across all packages
        cv_dates = set(si_cv_data['date'].values) & set(pvk_cv_data['date'].values)
        ramp_dates = set(si_ramp_data['date'].values) & set(pvk_ramp_data['date'].values)
        int_dates = set(si_int_data['date'].values) & set(pvk_int_data['date'].values)
        common_dates = sorted(list(cv_dates & ramp_dates & int_dates))
        
        if len(common_dates) == 0:
            print("[WARN] No common dates for combined analytical plot")
            return
        
        # Filter all datasets to common dates
        si_cv_filtered = si_cv_data[si_cv_data['date'].isin(common_dates)].sort_values('date')
        pvk_cv_filtered = pvk_cv_data[pvk_cv_data['date'].isin(common_dates)].sort_values('date')
        
        si_ramp_filtered = si_ramp_data[si_ramp_data['date'].isin(common_dates)].sort_values('date')
        pvk_ramp_filtered = pvk_ramp_data[pvk_ramp_data['date'].isin(common_dates)].sort_values('date')
        
        si_int_filtered = si_int_data[si_int_data['date'].isin(common_dates)].sort_values('date')
        pvk_int_filtered = pvk_int_data[pvk_int_data['date'].isin(common_dates)].sort_values('date')
        
        # Calculate differences for each package
        cv_actual_diff = si_cv_filtered['actual_cv'].values - pvk_cv_filtered['actual_cv'].values
        cv_pred_error_diff = si_cv_filtered['cv_error_absolute'].values - pvk_cv_filtered['cv_error_absolute'].values
        
        ramp_actual_diff = si_ramp_filtered['actual_mean_ramp'].values - pvk_ramp_filtered['actual_mean_ramp'].values
        ramp_pred_error_diff = si_ramp_filtered['ramp_error_mean_abs'].values - pvk_ramp_filtered['ramp_error_mean_abs'].values
        
        intermittency_actual_diff = si_int_filtered['actual_intermittency_ratio'].values - pvk_int_filtered['actual_intermittency_ratio'].values
        intermittency_pred_error_diff = si_int_filtered['intermittency_ratio_error'].values - pvk_int_filtered['intermittency_ratio_error'].values
        
        dates = pd.to_datetime(si_cv_filtered['date'].values)
        
        # Load weather data from evaluation results
        weather_df = self._load_weather_data_for_dates(common_dates)
        
        # Create 4-panel plot
        fig, axes = plt.subplots(4, 1, figsize=(15, 16))
        
        # Determine shared x-axis limits from weather data
        if weather_df is not None and len(weather_df) > 0:
            x_min = weather_df['timestamp'].min()
            x_max = weather_df['timestamp'].max()
        else:
            x_min = dates.min()
            x_max = dates.max()
        
        # Panel 1: Package 1 - Variability Analysis (CV)
        ax1 = axes[0]
        ax1.plot(dates, cv_actual_diff, 'purple', 
                label='Daily CV Actual Diff (Si - Pvk)', linewidth=2, alpha=0.8)
        ax1.plot(dates, cv_pred_error_diff, 'green', 
                label='Daily CV Prediction Error Diff (Si - Pvk)', linewidth=2, alpha=0.8)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax1.set_title('Package 1: Variability Analysis - CV Difference (Si vs Pvk)', 
                     fontsize=20, fontweight='bold')
        ax1.set_ylabel('CV Difference', fontsize=18, fontweight='bold')
        ax1.legend(fontsize=16, loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='both', which='major', labelsize=16)
        # Format x-axis with same settings as panel 4
        ax1.set_xlim(x_min, x_max)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.DayLocator(interval=14))
        
        # Panel 2: Package 2 - Ramp-Rate Analysis
        ax2 = axes[1]
        ax2.plot(dates, ramp_actual_diff, 'purple', 
                label='Daily Ramp Actual Diff (Si - Pvk)', linewidth=2, alpha=0.8)
        ax2.plot(dates, ramp_pred_error_diff, 'green', 
                label='Daily Ramp Prediction Error Diff (Si - Pvk)', linewidth=2, alpha=0.8)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.set_title('Package 2: Ramp-Rate Analysis - Ramp Rate Difference (Si vs Pvk)', 
                     fontsize=20, fontweight='bold')
        ax2.set_ylabel('Ramp Rate Difference', fontsize=18, fontweight='bold')
        ax2.legend(fontsize=16, loc='lower right')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='both', which='major', labelsize=16)
        # Format x-axis with same settings as panel 4
        ax2.set_xlim(x_min, x_max)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_locator(mdates.DayLocator(interval=14))
        
        # Panel 3: Package 3 - Intermittency Analysis
        ax3 = axes[2]
        ax3.plot(dates, intermittency_actual_diff, 'purple', 
                label='Daily Intermittency Actual Diff (Si - Pvk)', linewidth=2, alpha=0.8)
        ax3.plot(dates, intermittency_pred_error_diff, 'green', 
                label='Daily Intermittency Prediction Error Diff (Si - Pvk)', linewidth=2, alpha=0.8)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.set_title('Package 3: Intermittency Analysis - Intermittency Difference (Si vs Pvk)', 
                     fontsize=20, fontweight='bold')
        ax3.set_ylabel('Intermittency Ratio Difference', fontsize=18, fontweight='bold')
        ax3.legend(fontsize=16, loc='lower right')
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='both', which='major', labelsize=16)
        # Format x-axis with same settings as panel 4
        ax3.set_xlim(x_min, x_max)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax3.xaxis.set_major_locator(mdates.DayLocator(interval=14))
        
        # Panel 4: Weather Features
        ax4 = axes[3]
        
        if weather_df is not None and len(weather_df) > 0:
            # Plot weather features
            weather_features = ['TT_10', 'RF_10', 'RWS_10', 'RWS_IND_10', 'V_N']
            colors = ['purple', 'lightgreen', 'lightblue', 'cyan', 'brown']
            
            for feature, color in zip(weather_features, colors):
                if feature in weather_df.columns:
                    ax4.plot(weather_df['timestamp'].values, weather_df[feature].values, 
                            color=color, label=feature, linewidth=2, alpha=0.8)
            
            # Add irradiance on secondary y-axis
            if 'Irr' in weather_df.columns:
                ax4_secondary = ax4.twinx()
                irr_values = weather_df['Irr'].values
                ax4_secondary.plot(weather_df['timestamp'].values, irr_values, 
                                 'orange', label='Irr', linewidth=2, alpha=0.8)
                ax4_secondary.set_ylim(bottom=0, top=1200)
                ax4_secondary.set_ylabel('Irradiance (W/m²)', fontsize=18, fontweight='bold')
                ax4_secondary.tick_params(axis='y', which='major', labelsize=16)
                
                # Combine legends
                lines1, labels1 = ax4.get_legend_handles_labels()
                lines2, labels2 = ax4_secondary.get_legend_handles_labels()
                ax4.legend(lines1 + lines2, labels1 + labels2, fontsize=13, 
                          loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=6)
            else:
                ax4.legend(fontsize=13, loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=6)
            
            ax4.set_ylim(bottom=0, top=120)
        
        ax4.set_title('Weather Features', fontsize=20, fontweight='bold')
        ax4.set_xlabel('Date', fontsize=18, fontweight='bold')
        ax4.set_ylabel('Weather Values', fontsize=18, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='both', which='major', labelsize=16)
        
        # Format x-axis for dates (same as all panels above)
        ax4.set_xlim(x_min, x_max)
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax4.xaxis.set_major_locator(mdates.DayLocator(interval=14))
        
        plt.tight_layout()
        combined_pdf = report_dir / "analytical_packages_combined.pdf"
        plt.savefig(combined_pdf, format='pdf', bbox_inches='tight', facecolor='white')
        print(f"[INFO] Combined analytical packages plot (PDF) saved to: {combined_pdf}")
        plt.close(fig)
    
    def _load_weather_data_for_dates(self, dates: List) -> pd.DataFrame:
        """Load weather data for the given dates from evaluation results"""
        try:
            # Load weather-integrated data using the same method as other plots
            feats = []
            if self.preprocessor is not None:
                feats = list(self.preprocessor.feature_scalers.keys()) + list(self.preprocessor.output_scalers.keys())
            
            test_df = self.load_weather_integrated_data(feats, module_type="silicon")
            test_df = test_df.sort_values('timestamp')
            test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
            
            # Filter weather data for the given dates
            test_df['date'] = test_df['timestamp'].dt.date
            weather_df = test_df[test_df['date'].isin(dates)].copy()
            
            return weather_df
        except Exception as e:
            print(f"[WARN] Could not load weather data: {e}")
            return None

    def save_html_report(self, evaluation_results=None, analytical_results=None) -> str:
        """Erzeugt einen einzigen HTML-Report mit Plots und Metrik-Tabellen für beide Module-Typen"""
        import pandas as pd
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        report_dir = self.output_dir / self.model_name / timestamp
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / "evaluation_report.html"

        # Create model configuration table
        config_fig = self._create_config_table()
        config_table_html = ""
        if config_fig is not None:
            from plotly.io import to_html
            config_table_html = to_html(config_fig, include_plotlyjs='cdn', full_html=False)
        
        # Plots als HTML-DIVs
        try:
            feats = []
            if self.preprocessor is not None:
                feats = list(self.preprocessor.feature_scalers.keys()) + list(self.preprocessor.output_scalers.keys())
            
            # Load weather data for both module types
            self.loaded_test_df_si = self.load_weather_integrated_data(feats, module_type="silicon")
            self.loaded_test_df_si = self.loaded_test_df_si.sort_values('timestamp')
            
            self.loaded_test_df_pvk = self.load_weather_integrated_data(feats, module_type="perovskite")
            self.loaded_test_df_pvk = self.loaded_test_df_pvk.sort_values('timestamp')
            
            print(f"[INFO] Using entire weather datasets for report generation")
        except Exception as e:
            print(f"[WARN] Could not preload test data for weather subplot: {e}")
            self.loaded_test_df_si = None
            self.loaded_test_df_pvk = None
        plot_divs = self.create_visualizations_html(evaluation_results, analytical_results)

        # Metrics as table for both module types
        metrics_table = ""
        if evaluation_results:
            for module_type, module_data in evaluation_results.items():
                if module_data and 'metrics' in module_data:
                    module_metrics = module_data['metrics']
                    metrics_table += self._create_metrics_table(module_metrics, f"{module_type.upper()} Modules")
                    metrics_table += "<br><br>"


        # HTML-Report zusammenbauen
        html = f"""
        <html>
        <head>
            <title>Forecast Evaluation Report - {self.model_name}</title>
            <meta name='viewport' content='width=device-width, initial-scale=1.0'>
            <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
            <style>
                body {{
                    font-family: 'Inter', Arial, sans-serif;
                    margin: 40px;
                    background-color: #f5f5f5;
                    color: #22223b;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 10px;
                }}
                .metrics-table {{
                    border-collapse: collapse;
                    margin: 20px 0 30px 0;
                    width: 100%;
                    font-size: 12px;
                    background: white;
                    border-radius: 8px;
                    overflow: hidden;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
                }}
                .metrics-table th, .metrics-table td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                .metrics-table th {{
                    background-color: #3498db;
                    color: white;
                    font-weight: bold;
                }}
                .metrics-table tr:nth-child(even) {{
                    background-color: #f2f2f2;
                }}
                .plot-section {{
                    margin: 30px 0;
                    padding: 20px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    background: #f8fafc;
                }}
                .summary-box {{
                    background-color: #ecf0f1;
                    padding: 15px;
                    border-radius: 5px;
                    margin: 20px 0;
                }}
                .timestamp {{
                    color: #7f8c8d;
                    font-style: italic;
                    font-size: 0.98rem;
                }}
                .config-section {{
                    margin: 20px 0;
                    padding: 20px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    background: #f0f8ff;
                }}
                .analytical-package {{
                    margin: 30px 0;
                    padding: 20px;
                    border: 1px solid #27ae60;
                    border-radius: 8px;
                    background: #f8fff9;
                }}
                .transferability-summary {{
                    background: #fff5e6;
                    border-color: #e67e22;
                }}
                .insights-box {{
                    background-color: #ecf0f1;
                    padding: 15px;
                    border-radius: 5px;
                    margin: 15px 0;
                    border-left: 4px solid #3498db;
                }}
                .figure-section {{
                    margin: 20px 0;
                    padding: 15px;
                    background: #fafafa;
                    border-radius: 5px;
                    border: 1px solid #ddd;
                }}
                .figure-container {{
                    margin: 15px 0;
                    text-align: center;
                }}
                .figure-container h5 {{
                    margin-bottom: 10px;
                    color: #2c3e50;
                    font-size: 14px;
                }}
                @media (max-width: 800px) {{
                    .container {{
                        padding: 10px 2vw;
                    }}
                    .plot-section, .summary-box, .config-section {{
                        padding: 10px 2vw;
                    }}
                    .metrics-table {{
                        font-size: 0.95rem;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Forecast Evaluation Report</h1>
                <h2>Model: {self.model_name}</h2>
                <p class="timestamp"><b>Evaluation date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><b>Evaluation method:</b> {self.evaluator.__class__.__name__ if self.evaluator else 'Not loaded'}</p>

                <div class="config-section">
                    <h2>Model Configuration</h2>
                    {config_table_html if config_table_html else '<p>Configuration not available</p>'}
                </div>

                <div class="summary-box">
                    <h3>Evaluation Metrics</h3>
                    {metrics_table}
                </div>

                <div class="plot-section">
                    <h2>Predictions vs. Actual Values</h2>
                    <h3>Silicon Modules</h3>
                    {plot_divs.get('scatter_silicon','')}
                    <h3>Perovskite Modules</h3>
                    {plot_divs.get('scatter_perovskite','')}
                </div>
                <div class="plot-section">
                    <h2>Time Series Comparison (Silicon vs Perovskite)</h2>
                    {plot_divs.get('time_series','')}
                </div>
                <div class="plot-section">
                    <h2>Error Distribution</h2>
                    <h3>Silicon Modules</h3>
                    {plot_divs.get('error_dist_silicon','')}
                    <h3>Perovskite Modules</h3>
                    {plot_divs.get('error_dist_perovskite','')}
                </div>
                
                <!-- Analytical Packages Section -->
                {self._create_analytical_packages_section(analytical_results, plot_divs)}
                
            </div>
        </body>
        </html>
        """
        with open(report_path, 'w') as f:
            f.write(html)
        print(f"[INFO] HTML-Report saved: {report_path}")
        return str(report_path)
    
    def _create_error_distribution_plots_pdf(self, evaluation_results, report_dir):
        """Create error distribution plots in PDF format similar to experimental scenarios"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        if not evaluation_results:
            print("[WARN] No evaluation results available for error distribution plot generation")
            return
            
        for module_type, module_data in evaluation_results.items():
            if not module_data or 'predictions' not in module_data or 'actuals' not in module_data:
                continue
                
            predictions = module_data['predictions']
            actuals = module_data['actuals']
            
            # Handle multi-step predictions: flatten if needed
            if len(predictions.shape) == 3:  # Multi-step: (n_sequences, forecast_steps, n_features)
                print(f"[INFO] Flattening multi-step predictions for error distribution plot: {predictions.shape}")
                predictions_flat = predictions.reshape(-1, predictions.shape[-1])
                actuals_flat = actuals.reshape(-1, actuals.shape[-1])
            else:  # One-step: (n_points, n_features)
                predictions_flat = predictions
                actuals_flat = actuals
            
            # Create error distribution plot
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Calculate errors
            errors = predictions_flat[:, 0] - actuals_flat[:, 0]  # Assuming first output feature
            
            # Create histogram
            n, bins, patches = ax.hist(errors, bins=50, alpha=0.7, edgecolor='black', linewidth=0.5)
            
            # Color bars based on error magnitude
            for i, (bar_height, bin_left, bin_right) in enumerate(zip(n, bins[:-1], bins[1:])):
                bin_center = (bin_left + bin_right) / 2
                if abs(bin_center) < 0.05:  # Small errors
                    patches[i].set_facecolor('green')
                    patches[i].set_alpha(0.7)
                elif abs(bin_center) < 0.1:  # Medium errors
                    patches[i].set_facecolor('yellow')
                    patches[i].set_alpha(0.7)
                else:  # Large errors
                    patches[i].set_facecolor('red')
                    patches[i].set_alpha(0.7)
            
            # Add vertical line at zero
            ax.axvline(x=0, color='black', linestyle='--', linewidth=2, alpha=0.8)
            
            # Formatting - larger fonts, no statistics box
            ax.set_xlabel('Prediction Error (P_normalized)', fontsize=24, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=24, fontweight='bold')
            ax.set_title(f'{module_type.upper()} Modules - Error Distribution', 
                        fontsize=28, fontweight='bold', pad=20)
            ax.grid(True, alpha=0.3)
            
            # Increase tick label sizes - consistent with scatter plots
            ax.tick_params(axis='both', which='major', labelsize=20)
            
            # Tight layout
            plt.tight_layout()
            
            # Save plot
            plot_path = report_dir / f"error_distribution_{module_type}.pdf"
            plt.savefig(plot_path, format='pdf', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"[INFO] Error distribution plot saved: {plot_path}")
    
    def run_complete_evaluation(self) -> Dict[str, Any]:
        """Performs complete evaluation"""
        print("="*80)
        print(f"FORECAST EVALUATION FOR MODEL: {self.model_name}")
        print("="*80)
        
        try:
            # Load model first
            self.load_model()
            
            # Standard evaluation for both module types
            evaluation_results = self.run_standard_evaluation()
            
            # Analytical packages analysis
            analytical_results = self.run_analytical_packages(evaluation_results)
            
            # Walk-forward validation can be added later as a separate evaluator
            backtest_results = None
            
            # Visualizations
            # plot_paths = self.create_visualizations() # This line is removed as per the new_code
            
            # Save report with evaluation results and analytical results
            report_path = self.save_html_report(evaluation_results, analytical_results)
            
            # Generate PDF plots
            pdf_dir = self.save_pdf_plots(evaluation_results)
            
            # Generate analytical package PDF plots
            self.save_analytical_pdf_plots(analytical_results)
            
            # Summary
            results = {
                'model_name': self.model_name,
                'evaluation_results': evaluation_results,
                'report_path': report_path
            }
            results = {
                'model_name': self.model_name,
                'evaluation_results': evaluation_results
            }
            
            print("\n" + "="*80)
            print("OK Evaluation completed successfully for both module types!")
            print("="*80)
            
            return results
            
        except Exception as e:
            print(f"FAIL Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Forecast evaluation for LSTM models")
    parser.add_argument("--model-path", type=str, required=True, 
                       help="Path to model directory (e.g. results/trained_models/lstm/20250707_1152_lstm)")
    parser.add_argument("--silicon-data-path", type=str, default=None,
                       help="Path to Silicon validation dataset (optional, will auto-detect)")
    parser.add_argument("--perovskite-data-path", type=str, default=None,
                       help="Path to Perovskite validation dataset (optional, will auto-detect)")
    parser.add_argument("--output-dir", default="results/forecast_evaluation", help="Output directory")
    parser.add_argument("--exclude-periods", nargs='*', default=None,
                       help="Time periods to exclude from validation. Format: 'start_date,end_date' (e.g., '2024-01-01,2024-01-05' '2024-02-10,2024-02-15')")
    
    args = parser.parse_args()
    
    # Parse exclude periods
    exclude_periods = []
    if args.exclude_periods:
        for period_str in args.exclude_periods:
            try:
                start_date, end_date = period_str.split(',')
                exclude_periods.append((start_date.strip(), end_date.strip()))
            except ValueError:
                print(f"WARN: Invalid exclude period format: {period_str}. Expected: 'start_date,end_date'")
                continue
    
    # Validate model path
    model_path = Path(args.model_path)
    
    if not model_path.exists():
        print(f"Error: Model path not found: {model_path}")
        return
    
    print(f"Model path: {model_path}")
    print(f"Silicon data path: {args.silicon_data_path or 'auto-detect'}")
    print(f"Perovskite data path: {args.perovskite_data_path or 'auto-detect'}")
    print(f"Output directory: {args.output_dir}")
    if exclude_periods:
        print(f"Exclude periods: {exclude_periods}")
    else:
        print("Exclude periods: None")
    
    # Create evaluator and run evaluation
    try:
        evaluator = ForecastEvaluator(
            model_path=str(model_path),
            silicon_data_path=args.silicon_data_path,
            perovskite_data_path=args.perovskite_data_path,
            output_dir=args.output_dir,
            exclude_periods=exclude_periods
        )
        
        # Run complete evaluation
        results = evaluator.run_complete_evaluation()
        
        if results:
            print(f"OK Evaluation completed successfully!")
        else:
            print(f"FAIL Evaluation failed!")
            
    except Exception as e:
        print(f"FAIL Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Define paths here
    MODEL_PATH = ".../ML-forecasting/results/trained_models/lstm/20251019_2044_lstm"
    
    # Example: Use specific validation datasets
    # SILICON_DATA_PATH = "path/to/silicon_validation_data.csv"
    # PEROVSKITE_DATA_PATH = "path/to/perovskite_validation_data.csv"
    
    # Auto-detect validation datasets (will use training data if no specific datasets provided)
    # SILICON_DATA_PATH = "results/training_data/Silicon/cleanData/20250726_20250910/20250726_20250910_test_lstm_model_clean-10min_weather-integrated.csv"
    # PEROVSKITE_DATA_PATH = "results/training_data/Perovskite/cleanData/20250726_20250910/20250726_20250910_test_lstm_model_clean-10min_weather-integrated.csv"
    # EXCLUDE_PERIODS = [
    #     ("2025-08-24", "2025-09-02")
    # ]

    # SILICON_DATA_PATH = "results/training_data/Silicon/cleanData/20250716_20250914/20250716_20250914_test_lstm_model_clean-10min_weather-integrated.csv"
    # PEROVSKITE_DATA_PATH = "results/training_data/Perovskite/cleanData/20250716_20250914/20250716_20250914_test_lstm_model_clean-10min_weather-integrated.csv"
    # EXCLUDE_PERIODS = [
    #     ("2025-08-24", "2025-09-03"),
    #     ("2025-08-18", "2025-08-19"),
    # ]

    SILICON_DATA_PATH = "results/training_data/Silicon/cleanData/20241130_20241205/20241130_20241205_test_lstm_model_clean-10min_weather-integrated.csv"
    PEROVSKITE_DATA_PATH = "results/training_data/Perovskite/cleanData/20241130_20241205/20241130_20241205_test_lstm_model_clean-10min_weather-integrated.csv"
    EXCLUDE_PERIODS = [
    ]

    # SILICON_DATA_PATH = "results/training_data/Silicon/cleanData/20250316_20250328/20250316_20250328_test_lstm_model_clean-10min_weather-integrated.csv"
    # PEROVSKITE_DATA_PATH = "results/training_data/Perovskite/cleanData/20250316_20250328/20250316_20250328_test_lstm_model_clean-10min_weather-integrated.csv"
    # EXCLUDE_PERIODS = [
    #     ("2025-03-29", "2025-03-31")
    # ]
    
    OUTPUT_DIR = "results/forecast_evaluation"
    
    # Example: Define periods to exclude from validation
    # Format: List of tuples with (start_date, end_date) strings
    
    # Create evaluator and run evaluation
    try:
        evaluator = ForecastEvaluator(
            model_path=MODEL_PATH,
            silicon_data_path=SILICON_DATA_PATH,  # Auto-detect if None
            perovskite_data_path=PEROVSKITE_DATA_PATH,  # Auto-detect if None
            output_dir=OUTPUT_DIR,
            exclude_periods=EXCLUDE_PERIODS  # Exclude specified time periods
        )
        
        # Run complete evaluation for both module types
        results = evaluator.run_complete_evaluation()
        
        if results:
            print(f"OK Evaluation completed successfully for both module types!")
            print(f"Silicon results: {'OK' if 'silicon' in results.get('evaluation_results', {}) else 'FAIL'}")
            print(f"Perovskite results: {'OK' if 'perovskite' in results.get('evaluation_results', {}) else 'FAIL'}")
        else:
            print(f"FAIL Evaluation failed!")
            
    except Exception as e:
        print(f"FAIL Error during evaluation: {e}")
        import traceback
        traceback.print_exc() 
        traceback.print_exc() 