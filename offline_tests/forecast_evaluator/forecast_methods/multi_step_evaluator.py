#!/usr/bin/env python3
"""
Multi-Step-Ahead LSTM Model Evaluator

This module implements the direct multi-step-ahead evaluation approach
where the model predicts multiple future time steps in one forward pass.
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import torch

# Add project paths - will be determined dynamically based on model type
# Note: The correct path should be set by the ForecastEvaluator before this is called

# Import base class with fallback for direct execution
try:
    from ..base.base_evaluator import BaseEvaluator
except ImportError:
    from base.base_evaluator import BaseEvaluator


class MultiStepEvaluator(BaseEvaluator):
    """
    Multi-Step-Ahead LSTM Model Evaluator
    
    This class implements the direct multi-step-ahead evaluation approach
    where the model predicts multiple future time steps in one forward pass.
    """
    
    def __init__(self, model, preprocessor, config, model_name: str):
        """
        Initialize the Multi-Step Evaluator
        
        Args:
            model: Loaded LSTM model
            preprocessor: Loaded data preprocessor
            config: Model configuration dictionary
            model_name: Name of the model being evaluated
        """
        super().__init__(model, preprocessor, config, model_name)
        
        # Get forecast configuration
        forecast_mode = self.config.get("forecast_mode", {"mode": "multi-step", "forecast_steps": 1})
        self.forecast_steps = int(forecast_mode.get("forecast_steps", 1))
        
        self.evaluation_method = f'Multi-Step-Ahead (Direct, {self.forecast_steps} steps)'
        
        print(f"[INFO] MultiStepEvaluator initialized for model: {self.model_name}")
        print(f"[INFO] Forecast steps: {self.forecast_steps}")
        print(f"[INFO] Evaluation method: {self.evaluation_method}")
    
    def evaluate_module_type(self, test_data: pd.DataFrame, module_name: str) -> Dict[str, Any]:
        """
        Evaluate a specific module type using multi-step-ahead prediction
        
        Args:
            test_data: Test dataset for evaluation
            module_name: Name of the module type (e.g., "Silicon", "Perovskite")
            
        Returns:
            Dictionary containing evaluation results
        """
        print(f"\n[INFO] Starting Multi-Step evaluation for {module_name} modules")
        print(f"[INFO] Forecast steps: {self.forecast_steps}")
        
        # Import model classes dynamically using importlib.util
        import importlib.util
        
        # Determine model type and set correct path
        forecast_mode = self.config.get("forecast_mode", {"mode": "multi-step", "forecast_steps": 1})
        model_mode = forecast_mode.get("mode", "multi-step")
        
        if model_mode == "multi-step":
            model_path = Path(__file__).parent.parent.parent.parent / "lstm_model_multistep" / "service" / "src" / "model.py"
            preprocess_path = Path(__file__).parent.parent.parent.parent / "lstm_model_multistep" / "service" / "src" / "preprocess.py"
        else:
            model_path = Path(__file__).parent.parent.parent.parent / "lstm_model" / "service" / "src" / "model.py"
            preprocess_path = Path(__file__).parent.parent.parent.parent / "lstm_model" / "service" / "src" / "preprocess.py"
        
        # Import LSTMModel
        spec = importlib.util.spec_from_file_location("model", model_path)
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)
        LSTMModel = model_module.LSTMModel
        
        # Import DataPreprocessor
        spec = importlib.util.spec_from_file_location("preprocess", preprocess_path)
        preprocess_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(preprocess_module)
        DataPreprocessor = preprocess_module.DataPreprocessor
        
        # Check if this is a Weather-Informed model by checking the preprocessor
        weather_informed = hasattr(self.preprocessor, 'settings') and self.preprocessor.settings.get('weather_data', {}).get('use_weatherData', False)
        
        print(f"[DEBUG] Weather-Informed Detection:")
        print(f"[DEBUG]   hasattr(self.preprocessor, 'settings'): {hasattr(self.preprocessor, 'settings')}")
        if hasattr(self.preprocessor, 'settings'):
            print(f"[DEBUG]   self.preprocessor.settings: {self.preprocessor.settings}")
            weather_data = self.preprocessor.settings.get('weather_data', {})
            print(f"[DEBUG]   weather_data: {weather_data}")
            use_weatherData = weather_data.get('use_weatherData', False)
            print(f"[DEBUG]   use_weatherData: {use_weatherData}")
        print(f"[DEBUG]   weather_informed: {weather_informed}")
        
        if weather_informed:
            print(f"[INFO] Weather-Informed Multi-Step mode detected")
            print(f"[INFO] Using test data as weather forecast simulation")
            
            # For Weather-Informed mode, use the same data as both historical and forecast
            # This simulates having weather forecast data available during evaluation
            X_tensor, y_tensor = self.preprocessor.transform(test_data, weather_forecast_data=test_data)
        else:
            print(f"[INFO] Historical-only Multi-Step mode")
            # For historical-only mode, use standard transform
            X_tensor, y_tensor = self.preprocessor.transform(test_data)
        
        # Convert to numpy arrays
        X_scaled = X_tensor.numpy()
        y_scaled = y_tensor.numpy()
        
        # Create sequences for multi-step prediction
        sequences, targets = self._create_multistep_sequences(X_scaled, y_scaled)
        
        if len(sequences) == 0:
            raise ValueError(f"No valid sequences found for {module_name} evaluation")
        
        print(f"[INFO] Created {len(sequences)} sequences for multi-step evaluation")
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(sequences)
        y_tensor = torch.FloatTensor(targets)
        
        # Multi-step prediction
        self.model.eval()
        predictions = []
        actuals = []
        eval_timestamps = []
        
        with torch.no_grad():
            for i in range(len(X_tensor)):
                # Get sequence and target
                seq = X_tensor[i:i+1]  # Add batch dimension
                target = y_tensor[i]   # Shape: (forecast_steps, output_features)
                
                # Predict multiple steps at once
                pred = self.model(seq)  # Shape: (1, forecast_steps, output_features)
                pred = pred.squeeze(0)  # Remove batch dimension: (forecast_steps, output_features)
                
                # Store predictions and actuals
                predictions.append(pred.numpy())
                actuals.append(target.numpy())
                
                # Store timestamps for this prediction
                if weather_informed:
                    # For Weather-Informed mode, timestamps are calculated differently
                    # CORRECTION: In Weather-Informed mode, the sequence contains both historical and forecast data
                    # The target timestamps correspond to the forecast period after the historical part
                    # Since we're using step_size = forecast_steps (non-overlapping), the timestamp calculation is straightforward
                    # Sequence i contains: historical[i:i+8] + forecast[i+8:i+14]
                    # Target should be: timestamps[i+8:i+14]
                    start_idx = i + self.sequence_length + i * (self.forecast_steps - 1)
                    end_idx = start_idx + self.forecast_steps
                    
                    # print(f"[DEBUG] Weather-Informed Sequence {i}:")
                    # print(f"[DEBUG]   i = {i}, sequence_length = {self.sequence_length}, forecast_steps = {self.forecast_steps}")
                    # print(f"[DEBUG]   start_idx = {start_idx}, end_idx = {end_idx}")
                    # print(f"[DEBUG]   test_data length = {len(test_data)}")
                    
                    if end_idx <= len(test_data):
                        pred_timestamps = test_data['timestamp'].iloc[start_idx:end_idx].tolist()
                        eval_timestamps.extend(pred_timestamps)
                        # print(f"[DEBUG]   Predicted timestamps: {pred_timestamps}")
                        # print(f"[DEBUG]   First timestamp: {pred_timestamps[0]}")
                        # print(f"[DEBUG]   Last timestamp: {pred_timestamps[-1]}")
                        # print(f"[DEBUG]   Number of timestamps: {len(pred_timestamps)}")
                        
                        # Debug: Show what the model actually predicts vs what timestamps we assign
                        # print(f"[DEBUG]   Model prediction shape: {pred.numpy().shape}")
                        # print(f"[DEBUG]   Target shape: {target.numpy().shape}")
                        # print(f"[DEBUG]   Expected: {self.forecast_steps} timestamps, got: {len(pred_timestamps)}")
                    else:
                        print(f"[DEBUG]   ERROR: end_idx ({end_idx}) > test_data length ({len(test_data)})")
                        print(f"[DEBUG]   Skipping this sequence")
                else:
                    # For Historical-only mode, use non-overlapping logic
                    # Since we're using step_size = forecast_steps (non-overlapping), the timestamp calculation is straightforward
                    start_idx = i + self.sequence_length
                    end_idx = start_idx + self.forecast_steps
                    if end_idx <= len(test_data):
                        pred_timestamps = test_data['timestamp'].iloc[start_idx:end_idx].tolist()
                        eval_timestamps.extend(pred_timestamps)
                        print(f"[DEBUG] Historical Sequence {i}: Predicting timestamps {pred_timestamps[0]} to {pred_timestamps[-1]} ({len(pred_timestamps)} timestamps)")

        # Convert to numpy arrays
        predictions = np.array(predictions)  # Shape: (n_sequences, forecast_steps, output_features)
        actuals = np.array(actuals)          # Shape: (n_sequences, forecast_steps, output_features)
        
        # Debug shapes before flattening
        print(f"[INFO] Multi-step predictions shape: {predictions.shape}")
        print(f"[INFO] Multi-step actuals shape: {actuals.shape}")
        print(f"[INFO] Model output size: {self.model.output_size}")
        print(f"[INFO] Expected forecast steps: {self.forecast_steps}")
        
        # Check if model output matches expected forecast steps
        if predictions.shape[1] != self.forecast_steps:
            print(f"[WARN] Model output steps ({predictions.shape[1]}) don't match expected forecast steps ({self.forecast_steps})")
            print(f"[WARN] This suggests the model was not trained for multi-step prediction")
            print(f"[WARN] Adjusting evaluation to match model capabilities")
            
            # If model only outputs 1 step, we need to handle this differently
            if predictions.shape[1] == 1:
                # Repeat the single prediction for all forecast steps
                predictions_repeated = np.repeat(predictions, self.forecast_steps, axis=1)
                predictions = predictions_repeated
                print(f"[INFO] Repeated single prediction to match forecast steps: {predictions.shape}")
        else:
            print(f"[INFO] Model output matches expected forecast steps: {predictions.shape[1]} steps")
        
        # Reshape for metric calculation (flatten forecast steps)
        n_sequences, forecast_steps, n_outputs = predictions.shape
        predictions_flat = predictions.reshape(-1, n_outputs)  # (n_sequences * forecast_steps, n_outputs)
        actuals_flat = actuals.reshape(-1, n_outputs)          # (n_sequences * forecast_steps, n_outputs)
        
        # Inverse transform actuals back to original scale for comparison
        print(f"[DEBUG] Inverse transforming actuals back to original scale...")
        for i, output_feature in enumerate(self.output_features):
            if output_feature in self.preprocessor.output_scalers:
                scaler = self.preprocessor.output_scalers[output_feature]
                actuals_flat[:, i] = scaler.inverse_transform(actuals_flat[:, i].reshape(-1, 1)).flatten()
                print(f"[DEBUG] Inverse transformed {output_feature}: min={actuals_flat[:, i].min():.4f}, max={actuals_flat[:, i].max():.4f}")
        
        # Also inverse transform predictions for comparison
        print(f"[DEBUG] Inverse transforming predictions back to original scale...")
        for i, output_feature in enumerate(self.output_features):
            if output_feature in self.preprocessor.output_scalers:
                scaler = self.preprocessor.output_scalers[output_feature]
                predictions_flat[:, i] = scaler.inverse_transform(predictions_flat[:, i].reshape(-1, 1)).flatten()
                print(f"[DEBUG] Inverse transformed {output_feature} predictions: min={predictions_flat[:, i].min():.4f}, max={predictions_flat[:, i].max():.4f}")
        
        # Flatten timestamps to match flattened predictions
        # We need to collect timestamps in the SAME order as the flattened predictions
        timestamps_flat = []
        for i in range(n_sequences):
            if weather_informed:
                # For Weather-Informed mode, use NON-OVERLAPPING timestamp calculation
                # Since we use step_size = forecast_steps (non-overlapping), the calculation is different
                # For Weather-Informed: sequence i corresponds to timestamps starting at i * forecast_steps + sequence_length
                # But i here is the sequence index (0, 1, 2, 3, ...), not the data index (0, 6, 12, 18, ...)
                data_index = i * self.forecast_steps  # Convert sequence index to data index
                start_idx = data_index + self.sequence_length
                end_idx = start_idx + forecast_steps
            else:
                # For historical-only mode, use NON-OVERLAPPING calculation
                # Since we use step_size = forecast_steps (non-overlapping), the calculation is different
                # For Historical-only: sequence i corresponds to timestamps starting at i * forecast_steps + sequence_length
                # But i here is the sequence index (0, 1, 2, 3, ...), not the data index (0, 6, 12, 18, ...)
                data_index = i * self.forecast_steps  # Convert sequence index to data index
                start_idx = data_index + self.sequence_length
                end_idx = start_idx + forecast_steps
                
            if end_idx <= len(test_data):
                # Get the timestamps for this sequence's predictions
                seq_timestamps = test_data['timestamp'].iloc[start_idx:end_idx].tolist()
                timestamps_flat.extend(seq_timestamps)
                print(f"[DEBUG] Sequence {i}: Added {len(seq_timestamps)} timestamps from {seq_timestamps[0]} to {seq_timestamps[-1]}")
        
        print(f"[INFO] Total timestamps collected: {len(timestamps_flat)}")
        print(f"[INFO] First 5 timestamps: {timestamps_flat[:5]}")
        print(f"[INFO] Last 5 timestamps: {timestamps_flat[-5:]}")
        
        # Store timestamps for later synchronization (will be handled in forecast_evaluator.py)
        # No hardcoded timestamp shifting here - let the main evaluator handle synchronization
        
        # Since we now use NON-OVERLAPPING sequences, we don't need to handle duplicates
        # All predictions are already unique
        timestamps_unique = timestamps_flat
        predictions_unique = predictions_flat
        actuals_unique = actuals_flat
        
        print(f"[INFO] Non-overlapping predictions: {len(timestamps_unique)}")
        print(f"[INFO] First 5 timestamps: {timestamps_unique[:5]}")
        print(f"[INFO] Last 5 timestamps: {timestamps_unique[-5:]}")
        
        # Update the flattened arrays with unique data
        predictions_flat = predictions_unique
        actuals_flat = actuals_unique
        timestamps_flat = timestamps_unique
        
        print(f"[INFO] Flattened predictions shape: {predictions_flat.shape}")
        print(f"[INFO] Flattened actuals shape: {actuals_flat.shape}")
        print(f"[INFO] Flattened timestamps count: {len(timestamps_flat)}")
        print(f"[INFO] Expected timestamps count: {n_sequences * forecast_steps}")
        
        # Calculate metrics
        metrics = self._calculate_metrics(actuals_flat, predictions_flat)
        
        # Print detailed metrics for each output feature
        for feature in self.output_features:
            self._print_feature_metrics(module_name, feature, metrics)
        
        # Calculate overall metrics
        # Convert single metrics dict to per-feature format expected by _calculate_overall_metrics
        feature_metrics = {}
        for feature in self.output_features:
            feature_metrics[feature] = metrics
        
        overall_metrics = self._calculate_overall_metrics(feature_metrics)
        
        # Print overall assessment
        self._print_overall_assessment(module_name, overall_metrics)
        
        # Prepare evaluation results
        evaluation_results = {
            'model_name': self.model_name,
            'module_type': module_name,
            'evaluation_method': self.evaluation_method,
            'evaluation_timestamp': datetime.now().isoformat(),
            'forecast_steps': self.forecast_steps,
            'metrics': metrics,
            'data_points': len(actuals_flat),
            'sequence_length': self.sequence_length,
            'predictions': predictions_flat,  # Flattened for compatibility
            'actuals': actuals_flat,          # Flattened for compatibility
            'timestamps': timestamps_flat,  # Use flattened timestamps
            'predictions_multistep': predictions,  # Keep original multi-step shape
            'actuals_multistep': actuals           # Keep original multi-step shape
        }
        
        return evaluation_results
    
    def _create_multistep_sequences(self, X_scaled: np.ndarray, y_scaled: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for multi-step prediction
        
        For Weather-Informed models, X_scaled already contains the combined historical + forecast sequences.
        For Historical-only models, we create sequences using sliding window approach.
        
        Args:
            X_scaled: Scaled input features (already in sequence format for Weather-Informed)
            y_scaled: Scaled output targets
            
        Returns:
            Tuple of (sequences, targets) where targets have shape (n_sequences, forecast_steps, output_features)
        """
        # Check if X_scaled is already in sequence format (Weather-Informed mode)
        if len(X_scaled.shape) == 3:
            print(f"[INFO] Weather-Informed mode: X already in sequence format {X_scaled.shape}")
            print(f"[INFO] Using pre-created sequences from preprocessor")
            
            # X_scaled is already in the correct format: (n_sequences, sequence_length + forecast_steps, features)
            # For Weather-Informed mode, we use the full sequence as input (historical + forecast)
            # and create targets that correspond to the forecast period
            sequences = []
            targets = []
            
            # For Weather-Informed mode, each sequence contains historical + forecast data
            # The model expects the full sequence as input
            # Use NON-OVERLAPPING approach for clean evaluation
            step_size = self.forecast_steps  # Move by forecast_steps to create non-overlapping sequences
            for i in range(0, len(X_scaled), step_size):
                # Input sequence: full sequence (historical + forecast)
                seq = X_scaled[i]  # Shape: (sequence_length + forecast_steps, features)
                sequences.append(seq)
                
                # Target: create multi-step targets for the forecast period
                # For Weather-Informed mode, the target should correspond to the forecast period
                # which starts after the historical part of the sequence
                # Since we're using step_size = forecast_steps (non-overlapping), the target calculation is straightforward
                # For Weather-Informed: sequence i corresponds to targets starting at i + sequence_length
                target_start = i + self.sequence_length  # Start after historical part
                target_end = target_start + self.forecast_steps
                
                # print(f"[DEBUG] ===== WEATHER-INFORMED SEQUENCE {i} =====")
                # print(f"[DEBUG] Sequence Index: {i}")
                # print(f"[DEBUG] Sequence Length: {seq.shape[0]} (should be {self.sequence_length + self.forecast_steps})")
                # print(f"[DEBUG] Features per timestep: {seq.shape[1]}")
                # print(f"[DEBUG] Target Start Index: {target_start}")
                # print(f"[DEBUG] Target End Index: {target_end}")
                # print(f"[DEBUG] Available y_scaled length: {len(y_scaled)}")
                
                # # Show sequence structure
                # print(f"[DEBUG] Sequence Structure:")
                # print(f"[DEBUG]   Historical part (0-{self.sequence_length-1}): {seq[:self.sequence_length].shape}")
                # print(f"[DEBUG]   Forecast part ({self.sequence_length}-{seq.shape[0]-1}): {seq[self.sequence_length:].shape}")
                
                # Show sample values for each timestep
                # print(f"[DEBUG] Sample Values per Timestep:")
                # for t in range(min(3, seq.shape[0])):  # Show first 3 timesteps
                #     # print(f"[DEBUG]   Timestep {t}: {seq[t][:5]}... (first 5 features)")
                
                if target_end <= len(y_scaled):
                    target = y_scaled[target_start:target_end]  # Shape: (forecast_steps, output_features)
                    targets.append(target)
                    # print(f"[DEBUG] Target Shape: {target.shape}")
                    # print(f"[DEBUG] Target Values: {target.flatten()[:5]}... (first 5 values)")
                    # print(f"[DEBUG] Target Range: [{target.min():.4f}, {target.max():.4f}]")
                else:
                    # Not enough targets, skip this sequence
                    sequences.pop()  # Remove the last added sequence
                    # print(f"[DEBUG]   ERROR: target_end ({target_end}) > y_scaled length ({len(y_scaled)})")
                    # print(f"[DEBUG]   Skipping this sequence")
                    # print(f"[DEBUG]  2")
                    continue
                
                # print(f"[DEBUG] Weather-Informed Sequence {i}: Full sequence[{i}] -> Target[{target_start}:{target_end}]")
            
            print(f"[INFO] Created {len(sequences)} Weather-Informed sequences for multi-step evaluation")
            return np.array(sequences), np.array(targets)
        
        else:
            # Historical-only mode: create sequences using sliding window
            print(f"[INFO] Historical-only mode: creating sequences from 2D data")
            sequences = []
            targets = []
            
            # NON-OVERLAPPING approach: each sequence moves by forecast_steps for clean evaluation
            step_size = self.forecast_steps  # Move by forecast_steps to create non-overlapping sequences
            
            for i in range(0, len(X_scaled) - self.sequence_length - self.forecast_steps + 1, step_size):
                # Input sequence
                seq = X_scaled[i:i + self.sequence_length]
                sequences.append(seq)
                
                # Target sequence (next forecast_steps after the input sequence)
                # Since we're using step_size = forecast_steps (non-overlapping), the target calculation is straightforward
                # For Historical-only: sequence i corresponds to targets starting at i + sequence_length
                target_start = i + self.sequence_length
                target_end = target_start + self.forecast_steps
                target = y_scaled[target_start:target_end]
                targets.append(target)
                
                print(f"[DEBUG] ===== HISTORICAL-ONLY SEQUENCE {len(sequences)-1} =====")
                print(f"[DEBUG] Sequence Index: {i}")
                print(f"[DEBUG] Sequence Length: {seq.shape[0]} (should be {self.sequence_length})")
                print(f"[DEBUG] Features per timestep: {seq.shape[1]}")
                print(f"[DEBUG] Target Start Index: {target_start}")
                print(f"[DEBUG] Target End Index: {target_end}")
                print(f"[DEBUG] Available y_scaled length: {len(y_scaled)}")
                
                # Show sequence structure
                print(f"[DEBUG] Sequence Structure:")
                print(f"[DEBUG]   Input range: [{i}:{i+self.sequence_length}]")
                print(f"[DEBUG]   Target range: [{target_start}:{target_end}]")
                
                # Show sample values for each timestep
                print(f"[DEBUG] Sample Values per Timestep:")
                for t in range(min(3, seq.shape[0])):  # Show first 3 timesteps
                    print(f"[DEBUG]   Timestep {t}: {seq[t][:5]}... (first 5 features)")
                
                print(f"[DEBUG] Target Shape: {target.shape}")
                print(f"[DEBUG] Target Values: {target.flatten()[:5]}... (first 5 values)")
                print(f"[DEBUG] Target Range: [{target.min():.4f}, {target.max():.4f}]")
            
            print(f"[INFO] Created {len(sequences)} Historical-only sequences for multi-step evaluation")
            return np.array(sequences), np.array(targets)
    
    def _print_multistep_metrics(self, module_name: str, metrics: Dict[str, Dict[str, float]], output_features: List[str]):
        """
        Print multi-step specific metrics
        
        Args:
            module_name: Name of the module type
            metrics: Calculated metrics
            output_features: List of output feature names
        """
        print(f"\n[INFO] Multi-Step Metrics for {module_name}:")
        print(f"[INFO] Forecast horizon: {self.forecast_steps} steps")
        
        for feature in output_features:
            if feature in metrics:
                feature_metrics = metrics[feature]
                print(f"\n  {feature}:")
                print(f"    RMSE: {feature_metrics['rmse']:.6f}")
                print(f"    MAE:  {feature_metrics['mae']:.6f}")
                print(f"    MAPE: {feature_metrics['mape']:.2f}%")
                print(f"    R²:   {feature_metrics['r2']:.6f}")
                print(f"    Skill: {feature_metrics['skill_score']:.6f}")
    
    def _calculate_step_wise_metrics(self, actuals: np.ndarray, predictions: np.ndarray, output_features: List[str]) -> Dict[int, Dict[str, Dict[str, float]]]:
        """
        Calculate metrics for each forecast step separately
        
        Args:
            actuals: Actual values with shape (n_sequences, forecast_steps, output_features)
            predictions: Predicted values with shape (n_sequences, forecast_steps, output_features)
            output_features: List of output feature names
            
        Returns:
            Dictionary with step-wise metrics
        """
        step_metrics = {}
        
        for step in range(self.forecast_steps):
            step_actuals = actuals[:, step, :]  # (n_sequences, output_features)
            step_predictions = predictions[:, step, :]  # (n_sequences, output_features)
            
            step_metrics[step] = {}
            
            for i, feature in enumerate(output_features):
                actual_feature = step_actuals[:, i]
                pred_feature = step_predictions[:, i]
                
                # Calculate metrics for this step and feature
                rmse = np.sqrt(np.mean((pred_feature - actual_feature) ** 2))
                mae = np.mean(np.abs(pred_feature - actual_feature))
                
                # MAPE calculation with handling of zero values
                non_zero_mask = actual_feature != 0
                if np.any(non_zero_mask):
                    mape = np.mean(np.abs((actual_feature[non_zero_mask] - pred_feature[non_zero_mask]) / actual_feature[non_zero_mask])) * 100
                else:
                    mape = 0.0
                
                # R² calculation
                ss_res = np.sum((actual_feature - pred_feature) ** 2)
                ss_tot = np.sum((actual_feature - np.mean(actual_feature)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
                
                # Skill score (1 - RMSE/MAE of naive forecast)
                naive_mae = np.mean(np.abs(actual_feature - np.mean(actual_feature)))
                skill_score = 1 - (mae / naive_mae) if naive_mae != 0 else 0.0
                
                step_metrics[step][feature] = {
                    'rmse': rmse,
                    'mae': mae,
                    'mape': mape,
                    'r2': r2,
                    'skill_score': skill_score
                }
        
        return step_metrics
