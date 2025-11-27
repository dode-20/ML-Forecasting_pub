#!/usr/bin/env python3
"""
Base Evaluator Class

This module defines the abstract base class for all LSTM model evaluators.
It provides a common interface and shared functionality for different
evaluation approaches (one-step, multi-step, etc.).

"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime


class BaseEvaluator(ABC):
    """
    Abstract base class for all LSTM model evaluators
    
    This class defines the common interface that all evaluators must implement.
    It provides shared functionality and ensures consistency across different
    evaluation approaches.
    """
    
    def __init__(self, model, preprocessor, config, model_name: str):
        """
        Initialize the Base Evaluator
        
        Args:
            model: Loaded LSTM model
            preprocessor: Loaded data preprocessor
            config: Model configuration dictionary
            model_name: Name of the model being evaluated
        """
        self.model = model
        self.preprocessor = preprocessor
        self.config = config
        self.model_name = model_name
        
        # Get training settings
        self.training_settings = self.config.get('training_settings', {})
        self.sequence_length = self.training_settings.get("sequence_length", self.config.get("sequence_length", 48))
        self.output_features = self.training_settings.get("output", self.config.get("output", ["P_normalized"]))
        
        print(f"[INFO] {self.__class__.__name__} initialized for model: {self.model_name}")
        print(f"[INFO] Sequence length: {self.sequence_length}")
        print(f"[INFO] Output features: {self.output_features}")
    
    @abstractmethod
    def evaluate_module_type(self, test_data: pd.DataFrame, module_name: str) -> Dict[str, Any]:
        """
        Evaluate a specific module type using the specific evaluation approach
        
        Args:
            test_data: DataFrame with test data including timestamp column
            module_name: Name of the module type (e.g., "Silicon", "Perovskite")
            
        Returns:
            Dictionary containing evaluation results with the following structure:
            {
                'model_name': str,
                'module_type': str,
                'evaluation_method': str,
                'evaluation_timestamp': str,
                'metrics': dict,
                'data_points': int,
                'sequence_length': int,
                'predictions': np.ndarray,
                'actuals': np.ndarray,
                'timestamps': pd.Series
            }
        """
        pass
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive metrics for model evaluation
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            Dictionary containing calculated metrics
        """
        # Remove NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_true_clean) == 0:
            return {"error": "No valid data for metric calculation"}
        
        # Standard metrics
        mse = np.mean((y_true_clean - y_pred_clean) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true_clean - y_pred_clean))
        
        # MAPE with better handling of small values
        min_threshold = 0.1  # Only calculate MAPE for values > 0.1
        valid_indices = y_true_clean > min_threshold
        if np.sum(valid_indices) > 0:
            mape = np.mean(np.abs((y_true_clean[valid_indices] - y_pred_clean[valid_indices]) / y_true_clean[valid_indices])) * 100
        else:
            mape = 0.0
        
        # R²
        ss_res = np.sum((y_true_clean - y_pred_clean) ** 2)
        ss_tot = np.sum((y_true_clean - np.mean(y_true_clean)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        # Skill Score (against persistence model)
        persistence_pred = np.roll(y_true_clean, 1)
        persistence_pred[0] = y_true_clean[0]
        persistence_mse = np.mean((y_true_clean - persistence_pred) ** 2)
        skill_score = 1 - (mse / (persistence_mse + 1e-8))
        
        # Directional Accuracy
        y_true_diff = np.diff(y_true_clean)
        y_pred_diff = np.diff(y_pred_clean)
        directional_accuracy = np.mean((y_true_diff > 0) == (y_pred_diff > 0))
        
        # Peak Error
        peak_error = np.max(np.abs(y_true_clean - y_pred_clean))
        
        return {
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "r2": r2,
            "skill_score": skill_score,
            "directional_accuracy": directional_accuracy,
            "peak_error": peak_error,
            "data_points": len(y_true_clean)
        }
    
    def _print_feature_metrics(self, module_name: str, feature: str, feature_metrics: Dict[str, float]):
        """
        Print formatted metrics for a specific feature
        
        Args:
            module_name: Name of the module type
            feature: Name of the feature
            feature_metrics: Dictionary containing metrics for the feature
        """
        if feature == "P_normalized":
            print(f"[INFO] {module_name} Metrics for P_normalized (quantile-scaled values 0-1):")
            print(f"  - RMSE: {feature_metrics.get('rmse', 'N/A'):.4f} (normalized units)")
            print(f"  - MAE: {feature_metrics.get('mae', 'N/A'):.4f} (normalized units)")
            print(f"  - MAPE: {feature_metrics.get('mape', 'N/A'):.2f}% (relative error)")
            print(f"  - R²: {feature_metrics.get('r2', 'N/A'):.4f}")
            print(f"[INFO] To interpret: Lower RMSE/MAE = better prediction of normalized power values")
        else:
            print(f"  - RMSE: {feature_metrics.get('rmse', 'N/A'):.2f}")
            print(f"  - MAE: {feature_metrics.get('mae', 'N/A'):.2f}")
            print(f"  - MAPE: {feature_metrics.get('mape', 'N/A'):.2f}%")
            print(f"  - R²: {feature_metrics.get('r2', 'N/A'):.4f}")
        
        if "error" not in feature_metrics:
            print(f"  Skill Score: {feature_metrics['skill_score']:.4f}")
            print(f"  Directional Accuracy: {feature_metrics['directional_accuracy']:.2%}")
    
    def _calculate_overall_metrics(self, metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Calculate overall metrics across all features
        
        Args:
            metrics: Dictionary containing metrics for each feature
            
        Returns:
            Dictionary containing overall metrics
        """
        overall_metrics = {}
        for metric in ['rmse', 'mae', 'mape', 'r2', 'skill_score', 'directional_accuracy']:
            values = [feature_metrics[metric] for feature_metrics in metrics.values() if metric in feature_metrics]
            if values:
                overall_metrics[metric] = np.mean(values)
        
        return overall_metrics
    
    def _print_overall_assessment(self, module_name: str, overall_metrics: Dict[str, float]):
        """
        Print overall assessment for a module type
        
        Args:
            module_name: Name of the module type
            overall_metrics: Dictionary containing overall metrics
        """
        print(f"\n{module_name} OVERALL ASSESSMENT:")
        print(f"  Average RMSE: {overall_metrics.get('rmse', 0):.4f}")
        print(f"  Average MAE: {overall_metrics.get('mae', 0):.4f}")
        print(f"  Average MAPE: {overall_metrics.get('mape', 0):.2f}%")
        print(f"  Average R²: {overall_metrics.get('r2', 0):.4f}")
        print(f"  Average Skill Score: {overall_metrics.get('skill_score', 0):.4f}")
