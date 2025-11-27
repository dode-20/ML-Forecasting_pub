#!/usr/bin/env python3
"""
Model evaluation component for hyperparameter experiments.

Evaluates trained models using the existing forecast evaluator
and provides standardized metrics for parameter analysis.
"""

import sys
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import importlib.util
import tempfile
import shutil
import matplotlib.pyplot as plt
import numpy as np

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "offline_tests"))

import torch
from forecast_evaluator.base.evaluator_factory import EvaluatorFactory

class ExperimentModelEvaluator:
    """Evaluation component for hyperparameter experiments"""
    
    def __init__(self, evaluation_mode: str = "silicon_only", data_paths: Dict[str, Any] = None):
        """
        Initialize evaluator.
        
        Args:
            evaluation_mode: "silicon_only" or "cross_technology"
            data_paths: Data paths configuration
        """
        self.evaluation_mode = evaluation_mode
        self.data_paths = data_paths or {}
        self._load_forecast_evaluator()
    
    def _load_forecast_evaluator(self):
        """Load the existing forecast evaluator"""
        evaluator_path = Path(__file__).parent.parent.parent.parent / "forecast_evaluator" / "forecast_evaluator.py"
        print(f"DEBUG: Loading forecast evaluator from: {evaluator_path}")
        
        # Add forecast_evaluator directory to sys.path for imports
        forecast_evaluator_dir = evaluator_path.parent
        if str(forecast_evaluator_dir) not in sys.path:
            sys.path.insert(0, str(forecast_evaluator_dir))
        
        # CRITICAL FIX: Add lstm_model_multistep path to sys.path for correct model imports
        # This ensures multi_step_evaluator.py imports the correct multi-step model
        multistep_src_path = Path(__file__).parent.parent.parent.parent / "lstm_model_multistep" / "service" / "src"
        if str(multistep_src_path) not in sys.path:
            sys.path.insert(0, str(multistep_src_path))
            print(f"DEBUG: Added multi-step model path to sys.path: {multistep_src_path}")
        
        spec = importlib.util.spec_from_file_location("forecast_evaluator", evaluator_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.ForecastEvaluator = module.ForecastEvaluator
    
    def evaluate_silicon_only(self, model_path: Path, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate model performance on Silicon data only.
        
        Args:
            model_path: Path to trained model directory
            config: Model configuration
            
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"Evaluating Silicon-only performance: {model_path}")
        
        try:
            # Get validation data path
            self.model_path = model_path
            self.config = config
            self.silicon_data_path = self._get_validation_data_path("silicon")
            self.exclusion_periods = self._get_exclusion_periods()
            
            # Create temporary output directory for forecast evaluator
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_output_dir = Path(temp_dir) / "forecast_evaluation"
                
                # Initialize forecast evaluator
                print(f"DEBUG: Model path for ForecastEvaluator: {model_path}")
                print(f"DEBUG: Model path name: {model_path.name}")
                
                # Add debug info about what files exist in model directory
                print(f"DEBUG: Files in model directory:")
                for file in model_path.iterdir():
                    print(f"  - {file.name}")
                
                # Check if preprocessor file exists
                expected_preprocessor = model_path / f"preprocessor_{model_path.name.replace('_lstm', '')}_lstm.json"
                print(f"DEBUG: Looking for preprocessor file: {expected_preprocessor}")
                print(f"DEBUG: Preprocessor file exists: {expected_preprocessor.exists()}")
                
                # Also check what the ForecastEvaluator actually does
                print(f"DEBUG: Model path name: {model_path.name}")
                print(f"DEBUG: Model path name without '_lstm': {model_path.name.replace('_lstm', '')}")
                print(f"DEBUG: Expected preprocessor name: preprocessor_{model_path.name.replace('_lstm', '')}_lstm.json")
                
                print(f"DEBUG: About to initialize ForecastEvaluator...")
                evaluator = self.ForecastEvaluator(
                    model_path=str(model_path),
                    silicon_data_path=self.silicon_data_path,
                    perovskite_data_path=None,  # Not needed for silicon-only
                    output_dir=str(temp_output_dir),
                    exclude_periods=self.exclusion_periods
                )
                print(f"DEBUG: ForecastEvaluator initialized successfully: {type(evaluator)}")
                print(f"DEBUG: Evaluator has preprocessor: {hasattr(evaluator, 'preprocessor')}")
                if hasattr(evaluator, 'preprocessor'):
                    print(f"DEBUG: Preprocessor is None: {evaluator.preprocessor is None}")
                    if evaluator.preprocessor is not None:
                        print(f"DEBUG: Preprocessor has feature_scalers: {hasattr(evaluator.preprocessor, 'feature_scalers')}")
                        if hasattr(evaluator.preprocessor, 'feature_scalers'):
                            print(f"DEBUG: Feature scalers: {evaluator.preprocessor.feature_scalers}")
                        else:
                            print(f"DEBUG: Preprocessor does not have feature_scalers attribute!")
                    else:
                        print(f"DEBUG: Preprocessor is None!")
                else:
                    print(f"DEBUG: Evaluator does not have preprocessor attribute!")
                
                # self.load_model()
                
                # Run standard evaluation
                evaluator.load_model()
                evaluation_results = evaluator.run_complete_evaluation()
                
                # DEBUG: Check structure of evaluation_results
                print(f"DEBUG: evaluation_results keys: {list(evaluation_results.keys())}")
                if "evaluation_results" in evaluation_results:
                    print(f"DEBUG: evaluation_results['evaluation_results'] keys: {list(evaluation_results['evaluation_results'].keys())}")
                
                # DEBUG: Create plot to visualize actual vs predicted values
                actual_results = evaluation_results.get("evaluation_results", evaluation_results)
                self._create_debug_plot(actual_results, model_path)
                
                # Extract metrics from results - use the nested structure
                metrics = self._extract_metrics_from_results(actual_results, "silicon")
                
                return {
                    "evaluation_mode": "silicon_only",
                    "model_name": config["model_name"],
                    "metrics": metrics,
                    "raw_results": evaluation_results,
                    "success": True
                }
                
        except Exception as e:
            print(f"Evaluation failed: {e}")
            return {
                "evaluation_mode": "silicon_only",
                "model_name": config["model_name"],
                "metrics": {},
                "success": False,
                "error": str(e)
            }
    
    def evaluate_transferability(self, model_path: Path, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate cross-technology transferability (Si → Pvk).
        
        Args:
            model_path: Path to trained model directory
            config: Model configuration
            
        Returns:
            Dictionary with transferability metrics
        """
        print(f"Evaluating transferability: {model_path}")
        
        try:
            # Get validation data paths
            silicon_data_path = self._get_validation_data_path("silicon")
            perovskite_data_path = self._get_validation_data_path("perovskite")
            exclusion_periods = self._get_exclusion_periods()
            
            # Create temporary output directory for forecast evaluator
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_output_dir = Path(temp_dir) / "forecast_evaluation"
                
                # Initialize forecast evaluator
                evaluator = self.ForecastEvaluator(
                    model_path=str(model_path),
                    silicon_data_path=silicon_data_path,
                    perovskite_data_path=perovskite_data_path,
                    output_dir=str(temp_output_dir),
                    exclude_periods=exclusion_periods
                )
                
                # Run standard evaluation (evaluates both technologies)
                evaluator.load_model()
                evaluation_results = evaluator.run_standard_evaluation()
                
                # Extract metrics for both technologies
                silicon_metrics = self._extract_metrics_from_results(evaluation_results, "silicon")
                perovskite_metrics = self._extract_metrics_from_results(evaluation_results, "perovskite")
                
                # Calculate transferability metrics
                transfer_metrics = self._calculate_transfer_metrics(silicon_metrics, perovskite_metrics)
                
                return {
                    "evaluation_mode": "cross_technology",
                    "model_name": config["model_name"],
                    "silicon_metrics": silicon_metrics,
                    "perovskite_metrics": perovskite_metrics,
                    "transfer_metrics": transfer_metrics,
                    "raw_results": evaluation_results,
                    "success": True
                }
                
        except Exception as e:
            print(f"Transferability evaluation failed: {e}")
            return {
                "evaluation_mode": "cross_technology",
                "model_name": config["model_name"],
                "silicon_metrics": {},
                "perovskite_metrics": {},
                "transfer_metrics": {},
                "success": False,
                "error": str(e)
            }
    
    def _get_validation_data_path(self, module_type: str) -> str:
        """Get validation data path for module type"""
        validation_files = self.data_paths.get("validation_data_files", {})
        direct_file = validation_files.get(module_type)
        
        if direct_file and Path(direct_file).exists():
            return direct_file
        
        # Fallback to training data
        training_files = self.data_paths.get("training_data_files", {})
        fallback_file = training_files.get(module_type)
        
        if fallback_file and Path(fallback_file).exists():
            print(f"Using training data as validation data for {module_type}: {fallback_file}")
            return fallback_file
        
        raise FileNotFoundError(f"Validation data not found for {module_type}")
    
    def _get_exclusion_periods(self) -> List[Tuple[str, str]]:
        """Get exclusion periods from configuration"""
        exclusion_periods = self.data_paths.get("exclusion_periods", [])
        return [tuple(period) for period in exclusion_periods] if exclusion_periods else None
    
    def _extract_metrics_from_results(self, evaluation_results: Dict[str, Any], module_type: str) -> Dict[str, Any]:
        """Extract standardized metrics from forecast evaluator results"""
        metrics = {}
        
        print(f"DEBUG: Extracting metrics for {module_type}")
        print(f"DEBUG: Evaluation results keys: {list(evaluation_results.keys())}")
        
        # Look for module-specific results
        module_key = f"{module_type}_results"
        if module_key in evaluation_results:
            module_results = evaluation_results[module_key]
            print(f"DEBUG: Found {module_key}: {list(module_results.keys())}")
        elif module_type in evaluation_results:
            # Fallback: ForecastEvaluator uses direct module_type as key
            module_results = evaluation_results[module_type]
            print(f"DEBUG: Found {module_type}: {list(module_results.keys())}")
        
        if 'module_results' in locals():
            # Extract common metrics
            if "metrics" in module_results:
                metrics.update(module_results["metrics"])
                print(f"DEBUG: Extracted metrics: {metrics}")
            
            # Extract forecast-specific metrics
            if "forecast_metrics" in module_results:
                forecast_metrics = module_results["forecast_metrics"]
                forecast_dict = {
                    "RMSE": forecast_metrics.get("rmse", 0.0),
                    "MAE": forecast_metrics.get("mae", 0.0),
                    "R²": forecast_metrics.get("r2", 0.0),
                    "MAPE": forecast_metrics.get("mape", 0.0)
                }
                metrics.update(forecast_dict)
                print(f"DEBUG: Extracted forecast metrics: {forecast_dict}")
        
        # Look for direct metrics in evaluation_results
        if not metrics and "metrics" in evaluation_results:
            metrics = evaluation_results["metrics"]
            print(f"DEBUG: Found direct metrics: {metrics}")
        
        # Look for forecast_metrics in evaluation_results
        if not metrics and "forecast_metrics" in evaluation_results:
            forecast_metrics = evaluation_results["forecast_metrics"]
            metrics = {
                "RMSE": forecast_metrics.get("rmse", 0.0),
                "MAE": forecast_metrics.get("mae", 0.0),
                "R²": forecast_metrics.get("r2", 0.0),
                "MAPE": forecast_metrics.get("mape", 0.0)
            }
            print(f"DEBUG: Found direct forecast metrics: {metrics}")
        
        # If still no metrics, raise error instead of using fallback
        if not metrics:
            raise ValueError(f"No metrics found in evaluation results for {module_type}. Available keys: {list(evaluation_results.keys())}")
        
        return metrics
    
    def _calculate_transfer_metrics(self, silicon_results: Dict[str, Any], perovskite_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate transferability-specific metrics"""
        transfer_metrics = {}
        
        for metric in ["RMSE", "MAE", "R²", "MAPE"]:
            if metric in silicon_results and metric in perovskite_results:
                si_value = silicon_results[metric]
                pvk_value = perovskite_results[metric]
                
                # Calculate degradation (higher values = worse for RMSE, MAE, MAPE)
                if metric in ["RMSE", "MAE", "MAPE"]:
                    degradation = ((pvk_value - si_value) / si_value) * 100 if si_value > 0 else 0
                else:  # R² (higher is better)
                    degradation = ((si_value - pvk_value) / si_value) * 100 if si_value > 0 else 0
                
                transfer_metrics[f"{metric}_degradation_%"] = degradation
                transfer_metrics[f"{metric}_si"] = si_value
                transfer_metrics[f"{metric}_pvk"] = pvk_value
        
        return transfer_metrics
    
    def _create_debug_plot(self, evaluation_results: Dict[str, Any], model_path: Path):
        """Create debug plot to visualize actual vs predicted values"""
        try:
            print(f"DEBUG: Creating debug plot for model: {model_path.name}")
            
            # Extract silicon results
            if "silicon" not in evaluation_results:
                print("DEBUG: No silicon results found in evaluation_results")
                return
            
            silicon_results = evaluation_results["silicon"]
            
            # Extract predictions and actuals
            predictions = silicon_results.get("predictions")
            actuals = silicon_results.get("actuals")
            timestamps = silicon_results.get("timestamps")
            
            if predictions is None or actuals is None:
                print("DEBUG: Missing predictions or actuals in silicon results")
                return
            
            print(f"DEBUG: Predictions shape: {predictions.shape}")
            print(f"DEBUG: Actuals shape: {actuals.shape}")
            print(f"DEBUG: Timestamps length: {len(timestamps) if timestamps is not None else 'None'}")
            
            # Handle multi-step predictions: flatten if needed
            if len(predictions.shape) == 3:  # Multi-step: (n_sequences, forecast_steps, n_features)
                print(f"DEBUG: Flattening multi-step predictions: {predictions.shape}")
                predictions_flat = predictions.reshape(-1, predictions.shape[-1])
                actuals_flat = actuals.reshape(-1, actuals.shape[-1])
            else:  # One-step: (n_points, n_features)
                predictions_flat = predictions
                actuals_flat = actuals
            
            # Create debug plot with improved formatting
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Format model name for display
            model_display_name = model_path.name.replace('_', ' ').replace('baseline lstm model', 'LSTM Model')
            fig.suptitle(f'{model_display_name}\nActual vs Predicted Analysis', fontsize=18, fontweight='bold', y=0.95)
            
            # Plot 1: Time series comparison
            ax1 = axes[0, 0]
            time_indices = range(len(predictions_flat))
            ax1.plot(time_indices, actuals_flat[:, 0], 'b-', label='Actual', alpha=0.8, linewidth=1.5)
            ax1.plot(time_indices, predictions_flat[:, 0], 'r-', label='Predicted', alpha=0.8, linewidth=1.5)
            ax1.set_title('Time Series: Actual vs Predicted', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Time Index', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Power Output (normalized)', fontsize=12, fontweight='bold')
            ax1.legend(fontsize=11)
            ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            ax1.tick_params(axis='both', which='major', labelsize=10)
            
            # Plot 2: Scatter plot
            ax2 = axes[0, 1]
            ax2.scatter(actuals_flat[:, 0], predictions_flat[:, 0], alpha=0.6, s=8, color='blue', edgecolors='darkblue', linewidth=0.3)
            min_val = min(actuals_flat[:, 0].min(), predictions_flat[:, 0].min())
            max_val = max(actuals_flat[:, 0].max(), predictions_flat[:, 0].max())
            ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2.5, label='Perfect Prediction')
            ax2.set_title('Scatter: Predicted vs Actual', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Actual Power Output (normalized)', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Predicted Power Output (normalized)', fontsize=12, fontweight='bold')
            ax2.legend(fontsize=11)
            ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            ax2.tick_params(axis='both', which='major', labelsize=10)
            ax2.set_aspect('equal', adjustable='box')
            
            # Plot 3: Error over time
            ax3 = axes[1, 0]
            errors = predictions_flat[:, 0] - actuals_flat[:, 0]
            ax3.plot(time_indices, errors, 'g-', alpha=0.8, linewidth=1.5)
            ax3.axhline(y=0, color='r', linestyle='--', alpha=0.7, linewidth=2)
            ax3.set_title('Prediction Error Over Time', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Time Index', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Error (Predicted - Actual) [normalized]', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            ax3.tick_params(axis='both', which='major', labelsize=10)
            
            # Plot 4: Error distribution
            ax4 = axes[1, 1]
            ax4.hist(errors, bins=50, alpha=0.7, color='orange', edgecolor='black', linewidth=0.5)
            ax4.axvline(x=0, color='r', linestyle='--', alpha=0.7, linewidth=2)
            ax4.set_title('Error Distribution', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Error (Predicted - Actual) [kW]', fontsize=12, fontweight='bold')
            ax4.set_ylabel('Frequency', fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            ax4.tick_params(axis='both', which='major', labelsize=10)
            
            # Add statistics text with improved formatting
            mae = np.mean(np.abs(errors))
            rmse = np.sqrt(np.mean(errors**2))
            r2 = 1 - np.sum(errors**2) / np.sum((actuals_flat[:, 0] - np.mean(actuals_flat[:, 0]))**2)
            
            stats_text = f'MAE: {mae:.4f} kW\nRMSE: {rmse:.4f} kW\nR²: {r2:.4f}'
            fig.text(0.02, 0.02, stats_text, fontsize=12, fontweight='bold', 
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.9))
            
            plt.tight_layout()
            
            # Save PNG version (high resolution)
            debug_plot_png = model_path / f"debug_plot_{model_path.name}.png"
            plt.savefig(debug_plot_png, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"DEBUG: Debug plot (PNG) saved to: {debug_plot_png}")
            
            # Save PDF version (vector format)
            debug_plot_pdf = model_path / f"debug_plot_{model_path.name}.pdf"
            plt.savefig(debug_plot_pdf, format='pdf', bbox_inches='tight', facecolor='white')
            print(f"DEBUG: Debug plot (PDF) saved to: {debug_plot_pdf}")
            
            # Save individual subplots as separate PDFs
            subplot_titles = [
                "time_series_comparison",
                "scatter_plot", 
                "error_over_time",
                "error_distribution"
            ]
            
            for i, (ax, title) in enumerate(zip(axes.flat, subplot_titles)):
                # Create individual figure for each subplot
                fig_individual, ax_individual = plt.subplots(1, 1, figsize=(10, 8))
                
                # Copy the subplot content
                if i == 0:  # Time series
                    ax_individual.plot(time_indices, actuals_flat[:, 0], 'b-', label='Actual', alpha=0.8, linewidth=2.5)
                    ax_individual.plot(time_indices, predictions_flat[:, 0], 'r-', label='Predicted', alpha=0.8, linewidth=2.5)
                    ax_individual.set_title('Time Series: Actual vs Predicted', fontsize=24, fontweight='bold')
                    ax_individual.set_xlabel('Time Index', fontsize=20, fontweight='bold')
                    ax_individual.set_ylabel('Power Output (normalized)', fontsize=20, fontweight='bold')
                    ax_individual.legend(fontsize=18)
                    ax_individual.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                    ax_individual.tick_params(axis='both', which='major', labelsize=16)
                    
                elif i == 1:  # Scatter plot
                    ax_individual.scatter(actuals_flat[:, 0], predictions_flat[:, 0], alpha=0.6, s=20, color='blue', edgecolors='darkblue', linewidth=0.5)
                    min_val = min(actuals_flat[:, 0].min(), predictions_flat[:, 0].min())
                    max_val = max(actuals_flat[:, 0].max(), predictions_flat[:, 0].max())
                    ax_individual.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=4, label='Perfect Prediction')
                    ax_individual.set_title('Scatter: Predicted vs Actual', fontsize=24, fontweight='bold')
                    ax_individual.set_xlabel('Actual Power Output (normalized)', fontsize=20, fontweight='bold')
                    ax_individual.set_ylabel('Predicted Power Output (normalized)', fontsize=20, fontweight='bold')
                    ax_individual.legend(fontsize=18)
                    ax_individual.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                    ax_individual.tick_params(axis='both', which='major', labelsize=16)
                    ax_individual.set_aspect('equal', adjustable='box')
                    
                elif i == 2:  # Error over time
                    errors = predictions_flat[:, 0] - actuals_flat[:, 0]
                    ax_individual.plot(time_indices, errors, 'g-', alpha=0.8, linewidth=2.5)
                    ax_individual.axhline(y=0, color='r', linestyle='--', alpha=0.7, linewidth=3)
                    ax_individual.set_title('Prediction Error Over Time', fontsize=24, fontweight='bold')
                    ax_individual.set_xlabel('Time Index', fontsize=20, fontweight='bold')
                    ax_individual.set_ylabel('Error (Predicted - Actual) [normalized]', fontsize=20, fontweight='bold')
                    ax_individual.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                    ax_individual.tick_params(axis='both', which='major', labelsize=16)
                    
                elif i == 3:  # Error distribution
                    errors = predictions_flat[:, 0] - actuals_flat[:, 0]
                    ax_individual.hist(errors, bins=50, alpha=0.7, color='orange', edgecolor='black', linewidth=0.8)
                    ax_individual.axvline(x=0, color='r', linestyle='--', alpha=0.7, linewidth=3)
                    ax_individual.set_title('Error Distribution', fontsize=24, fontweight='bold')
                    ax_individual.set_xlabel('Error (Predicted - Actual) [normalized]', fontsize=20, fontweight='bold')
                    ax_individual.set_ylabel('Frequency', fontsize=20, fontweight='bold')
                    ax_individual.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                    ax_individual.tick_params(axis='both', which='major', labelsize=16)
                
                # Add model name as subtitle
                ax_individual.text(0.5, 0.95, model_display_name, transform=ax_individual.transAxes, 
                                fontsize=18, ha='center', va='top', 
                                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
                
                plt.tight_layout()
                
                # Save individual subplot as PDF
                subplot_pdf = model_path / f"debug_plot_{model_path.name}_{title}.pdf"
                plt.savefig(subplot_pdf, format='pdf', bbox_inches='tight', facecolor='white')
                print(f"DEBUG: Subplot {title} (PDF) saved to: {subplot_pdf}")
                
                plt.close(fig_individual)
            
            plt.close()
            
            # Print additional debug info
            print(f"DEBUG: Actual values - min: {actuals_flat[:, 0].min():.4f}, max: {actuals_flat[:, 0].max():.4f}, mean: {actuals_flat[:, 0].mean():.4f}")
            print(f"DEBUG: Predicted values - min: {predictions_flat[:, 0].min():.4f}, max: {predictions_flat[:, 0].max():.4f}, mean: {predictions_flat[:, 0].mean():.4f}")
            print(f"DEBUG: Error statistics - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
            
        except Exception as e:
            print(f"DEBUG: Failed to create debug plot: {e}")
            import traceback
            traceback.print_exc()
    
    def evaluate_model(self, model_path: Path, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate model based on configured evaluation mode.
        
        Args:
            model_path: Path to trained model directory
            config: Model configuration
            
        Returns:
            Dictionary with evaluation results
        """
        if self.evaluation_mode == "silicon_only":
            return self.evaluate_silicon_only(model_path, config)
        elif self.evaluation_mode == "cross_technology":
            return self.evaluate_transferability(model_path, config)
        else:
            raise ValueError(f"Unknown evaluation mode: {self.evaluation_mode}")


    def load_model(self):
        """Load the specified model"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path not found: {self.model_path}")
        
        # Extract model name from path
        self.model_name = self.model_path.name.replace('_lstm', '')
        print(f"[INFO] Loading model from: {self.model_path}")
        
        # Determine model type and set correct path
        forecast_mode = self.config.get("forecast_mode", {"mode": "one-step", "forecast_steps": 1})
        model_mode = forecast_mode.get("mode", "one-step")
        
        if model_mode == "multi-step":
            src_path = Path(__file__).parent.parent.parent.parent / "lstm_model_multistep" / "service" / "src"
            print(f"[INFO] Using Multi-Step model path: {src_path}")
        else:
            src_path = Path(__file__).parent.parent.parent.parent / "lstm_model" / "service" / "src"
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
            preprocess_path = Path(__file__).parent.parent.parent.parent.parent / "lstm_model_multistep" / "service" / "src" / "preprocess.py"
            spec = importlib.util.spec_from_file_location("multistep_preprocess", preprocess_path)
            preprocess_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(preprocess_module)
            DataPreprocessor = preprocess_module.DataPreprocessor
            print(f"[INFO] Using DataPreprocessor from lstm_model_multistep for multi-step model")
        else:
            # Use DataPreprocessor from lstm_model (one-step)
            preprocess_path = Path(__file__).parent.parent.parent.parent.parent / "lstm_model" / "service" / "src" / "preprocess.py"
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
            
        # Import LSTMModel here to avoid circular imports
        import importlib.util
        
        if model_mode == "multi-step":
            model_path = Path(__file__).parent.parent.parent.parent.parent / "lstm_model_multistep" / "service" / "src" / "model.py"
            spec = importlib.util.spec_from_file_location("multistep_model", model_path)
            model_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(model_module)
            LSTMModel = model_module.LSTMModel
        else:
            model_path = Path(__file__).parent.parent.parent.parent.parent / "lstm_model" / "service" / "src" / "model.py"
            spec = importlib.util.spec_from_file_location("onestep_model", model_path)
            model_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(model_module)
            LSTMModel = model_module.LSTMModel
        
        # Determine forecast_steps for model initialization
        forecast_steps = 1  # Default for one-step models
        if "forecast_mode" in self.config:
            forecast_mode = self.config["forecast_mode"]
            if isinstance(forecast_mode, dict) and forecast_mode.get("mode") == "multi-step":
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
        self.model.load_state_dict(torch.load(model_file, map_location=torch.device("cpu")))
        self.model.eval()
        
        # Create appropriate evaluator based on model type
        self.evaluator = EvaluatorFactory.create_evaluator(
            self.model, self.preprocessor, self.config, self.model_name
        )
        
        print(f"[OK] Model {self.model_name} loaded successfully")
        print(f"[OK] Using evaluator: {self.evaluator.__class__.__name__}")
        print(f"[DEBUG] Evaluator has run_standard_evaluation: {hasattr(self.evaluator, 'run_standard_evaluation')}")
        print(f"[DEBUG] Evaluator methods: {[m for m in dir(self.evaluator) if not m.startswith('_')]}")
        return ALL_FEATURES