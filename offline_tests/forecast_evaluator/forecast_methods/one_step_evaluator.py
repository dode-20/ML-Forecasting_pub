#!/usr/bin/env python3
"""
One-Step-Ahead LSTM Evaluator

This module implements the iterative one-step-ahead evaluation approach
for LSTM models. It extracts the core evaluation logic from the main
forecast_evaluator.py and makes it reusable as a separate class.

"""

import sys
import os
import json
import pandas as pd
import numpy as np
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


class OneStepEvaluator(BaseEvaluator):
    """
    One-Step-Ahead LSTM Model Evaluator
    
    This class implements the iterative one-step-ahead evaluation approach
    where the model predicts one step at a time and uses actual data for
    the next prediction step.
    """
    
    def __init__(self, model, preprocessor, config, model_name: str):
        """
        Initialize the One-Step Evaluator
        
        Args:
            model: Loaded LSTM model
            preprocessor: Loaded data preprocessor
            config: Model configuration dictionary
            model_name: Name of the model being evaluated
        """
        super().__init__(model, preprocessor, config, model_name)
        print(f"[INFO] OneStepEvaluator initialized for model: {self.model_name}")
    
    def evaluate_module_type(self, test_data: pd.DataFrame, module_name: str) -> Dict[str, Any]:
        """
        Evaluate a specific module type using iterative one-step-ahead prediction
        
        Args:
            test_data: DataFrame with test data including timestamp column
            module_name: Name of the module type (e.g., "Silicon", "Perovskite")
            
        Returns:
            Dictionary containing evaluation results
        """
        print(f"[INFO] Evaluating {module_name} modules with One-Step-Ahead approach...")
        
        # Import model classes dynamically (path should be set by ForecastEvaluator)
        from model import LSTMModel
        from preprocess import DataPreprocessor
        
        # Store timestamps for visualization (full series from data file)
        timestamps = test_data['timestamp']
        
        # Remove timestamp column for processing
        test_data_processed = test_data.drop('timestamp', axis=1, errors='ignore')
        
        # Prepare data
        if test_data_processed.isnull().any().any():
            print(f"[WARN] NaN values found in {module_name} data, replacing with 0")
            test_data_processed = test_data_processed.fillna(0)

        # DEBUG: P-Statistiken vor Transform (Roh-Werte in Watt)
        if 'P' in test_data.columns:
            try:
                p_series = pd.to_numeric(test_data['P'], errors='coerce')
                print(f"[DEBUG] {module_name} Pre-transform 'P' stats → min: {p_series.min():.4f}, max: {p_series.max():.4f}, mean: {p_series.mean():.4f}")
            except Exception as e:
                print(f"[WARN] Could not compute pre-transform 'P' stats for {module_name}: {e}")
        
        # DEBUG: Show what data is being transformed
        print(f"[DEBUG] {module_name} Data columns before transform: {list(test_data_processed.columns)}")
        print(f"[DEBUG] {module_name} Data shape before transform: {test_data_processed.shape}")
        
        # Transform data
        X, y = self.preprocessor.transform(test_data_processed)
        
        # DEBUG: Show what was actually transformed
        print(f"[DEBUG] {module_name} X (input features) shape: {X.shape}")
        print(f"[DEBUG] {module_name} y (output features) shape: {y.shape}")
        print(f"[DEBUG] {module_name} X contains features: {list(self.preprocessor.feature_scalers.keys())}")
        print(f"[DEBUG] {module_name} y contains features: {list(self.preprocessor.output_scalers.keys())}")
        
        # DEBUG: y (scaled) Statistiken
        try:
            print(f"[DEBUG] {module_name} y (scaled) shape: {getattr(y, 'shape', None)} → min: {np.min(y):.6f}, max: {np.max(y):.6f}, mean: {np.mean(y):.6f}")
        except Exception as e:
            print(f"[WARN] Could not compute stats for {module_name} y (scaled): {e}")
        
        # Create sequences for initial prediction
        X_seq, y_seq = self.preprocessor.create_sequences(X, y, sequence_length=self.sequence_length)
        
        # DEBUG: y_seq (scaled) statistics and inverse transformation test
        try:
            print(f"[DEBUG] {module_name} y_seq (scaled) shape: {getattr(y_seq, 'shape', None)}")
            if len(y_seq) > 0:
                sample_inv = self.preprocessor.inverse_transform_output(y_seq[0:1])
                print(f"[DEBUG] {module_name} y_seq[0] inverse-transformed (first 5): {np.array(sample_inv).flatten()[:5]}")
        except Exception as e:
            print(f"[WARN] Could not inverse-transform {module_name} y_seq sample: {e}")
        
        # Run iterative one-step-ahead prediction
        print(f"[INFO] Running iterative one-step-ahead prediction for {module_name} with {len(X_seq)} sequences...")
        self.model.eval()
        
        predictions = []
        actuals = []
        align_debug = []
        
        with torch.no_grad():
            for i in range(len(X_seq)):
                # Get current sequence
                current_seq = X_seq[i:i+1]  # Shape: (1, sequence_length, features)
                
                # Make prediction (ONE step ahead)
                pred_scaled = self.model(current_seq)
                
                # Inverse transform prediction
                pred = self.preprocessor.inverse_transform_output(pred_scaled)
                
                # Store results
                if hasattr(pred, 'numpy'):
                    predictions.append(pred.numpy().flatten())
                else:
                    predictions.append(pred.flatten())
                
                # Inverse transform actual values too (they are scaled!)
                actual = self.preprocessor.inverse_transform_output(y_seq[i:i+1])
                if hasattr(actual, 'numpy'):
                    actuals.append(actual.numpy().flatten())
                else:
                    actuals.append(actual.flatten())
                
                if (i + 1) % 100 == 0:
                    print(f"[INFO] Completed {i + 1}/{len(X_seq)} predictions for {module_name}")

                # Alignment debug: target timestamp for this step is timestamps[i+sequence_length]
                ts_idx = i + self.sequence_length
                ts_val = timestamps.iloc[ts_idx] if (ts_idx < len(timestamps)) else None
                try:
                    align_debug.append((ts_idx, ts_val, float(pred.flatten()[0]), float(actual.flatten()[0])))
                except Exception:
                    pass
        
        # Convert to numpy arrays
        predictions = np.array(predictions)
        actuals = np.array(actuals)

        # Align timestamps to prediction/target index (shift by sequence_length)
        try:
            start_idx = self.sequence_length-1
            end_idx = start_idx + len(actuals)
            eval_timestamps = timestamps.iloc[start_idx:end_idx].reset_index(drop=True)
        except Exception:
            eval_timestamps = timestamps[:len(actuals)]
        
        # DEBUG: Check actual values and predictions + alignment
        try:
            for i, feature in enumerate(self.output_features):
                print(f"[DEBUG] {module_name} {feature.upper()} - First 10 actual values: {actuals[:10, i]}")
                print(f"[DEBUG] {module_name} {feature.upper()} - First 10 predictions: {predictions[:10, i]}")
                print(f"[DEBUG] {module_name} {feature.upper()} - Actual values - min: {actuals[:, i].min():.4f}, max: {actuals[:, i].max():.4f}, mean: {actuals[:, i].mean():.4f}")
                print(f"[DEBUG] {module_name} {feature.upper()} - Predictions - min: {predictions[:, i].min():.4f}, max: {predictions[:, i].max():.4f}, mean: {predictions[:, i].mean():.4f}")
                
                # Special handling for P_normalized
                if feature == "P_normalized":
                    print(f"[INFO] {module_name} P_normalized values are quantile-scaled (0-1 range). Original P values would be: P = P_normalized * scaling_factor")
                    print(f"[INFO] To convert back to original P values, multiply by the scaling factor from the model directory")
            
            if align_debug:
                print(f"[DEBUG] {module_name} Alignment samples (idx, timestamp, pred, actual):")
                samples = [0, len(align_debug)//2, len(align_debug)-1]
                for k in samples:
                    idx, ts, pv, av = align_debug[k]
                    print(f"  i={idx} | ts={ts} | pred={pv:.4f} | actual={av:.4f}")
        except Exception as e:
            print(f"[WARN] Could not print debug stats for {module_name} actuals/predictions: {e}")
        
        # Calculate metrics for each output variable
        metrics = {}
        
        for i, feature in enumerate(self.output_features):
            print(f"\n{module_name} {feature.upper()}:")
            feature_metrics = self._calculate_metrics(actuals[:, i], predictions[:, i])
            metrics[feature] = feature_metrics
            
            # Use base class method for consistent formatting
            self._print_feature_metrics(module_name, feature, feature_metrics)
        
        # Calculate overall metrics using base class method
        overall_metrics = self._calculate_overall_metrics(metrics)
        metrics['overall'] = overall_metrics
        
        evaluation_results = {
            'model_name': self.model_name,
            'module_type': module_name,
            'evaluation_method': 'One-Step-Ahead (Iterative)',
            'evaluation_timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'data_points': len(actuals),
            'sequence_length': self.sequence_length,
            'predictions': predictions,
            'actuals': actuals,
            'timestamps': eval_timestamps
        }
        
        # Use base class method for consistent formatting
        self._print_overall_assessment(module_name, overall_metrics)
        
        return evaluation_results
