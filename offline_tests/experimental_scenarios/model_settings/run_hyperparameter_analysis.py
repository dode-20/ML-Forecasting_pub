#!/usr/bin/env python3
"""
Hyperparameter Analysis Runner

Systematically analyzes the impact of different hyperparameters on LSTM model performance.
Supports both Silicon-only optimization and cross-technology transferability analysis.

Usage:
    python run_hyperparameter_analysis.py --mode silicon_only --parameters all
    python run_hyperparameter_analysis.py --mode cross_technology --parameters epochs,batch_size
    python run_hyperparameter_analysis.py --mode silicon_only --parameters learning_rate --config custom_config.json
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.config_manager import ConfigManager
from core.evaluation_modes import EvaluationModes
from src.model_trainer import ExperimentModelTrainer
from src.model_evaluator import ExperimentModelEvaluator
from src.report_generator import HyperparameterReportGenerator

class HyperparameterAnalysis:
    """Main class for hyperparameter analysis experiments"""
    
    def __init__(self, evaluation_mode: str = "silicon_only", base_config: str = "baseline_lstm_model_config.json"):
        """
        Initialize hyperparameter analysis.
        
        Args:
            evaluation_mode: "silicon_only" or "cross_technology"
            base_config: Base configuration file name (in configs/ directory)
        """
        self.evaluation_mode = evaluation_mode
        self.mode_config = EvaluationModes.get_mode(evaluation_mode)
        
        # Initialize components (base_config is now relative to configs directory)
        self.config_manager = ConfigManager(base_config)
        
        # Setup output directories
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = Path(__file__).parent / "experiments" / f"{evaluation_mode}_{self.timestamp}"
        self.trained_models_dir = self.experiment_dir / "trained_models"
        self.reports_dir = self.experiment_dir / "reports"
        
        # Create directories
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.trained_models_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data paths configuration
        self.data_paths = self._load_data_paths()
        
        # Initialize trainer and evaluator
        self.trainer = ExperimentModelTrainer(self.trained_models_dir)
        self.evaluator = ExperimentModelEvaluator(evaluation_mode, self.data_paths)
        self.report_generator = HyperparameterReportGenerator(self.reports_dir)
        
        print(f"Hyperparameter Analysis initialized:")
        print(f"  Mode: {evaluation_mode}")
        print(f"  Base config: {base_config}")
        print(f"  Experiment directory: {self.experiment_dir}")
    
    def _load_data_paths(self) -> Dict[str, Any]:
        """Load data paths configuration"""
        config_path = Path(__file__).parent / "configs" / "data_paths.json"
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def load_parameter_ranges(self) -> Dict[str, Any]:
        """Load parameter ranges configuration"""
        ranges_path = Path(__file__).parent / "configs" / "parameter_ranges.json"
        with open(ranges_path, 'r') as f:
            return json.load(f)
    
    def analyze_parameter(self, parameter_name: str) -> Dict[str, Any]:
        """
        Analyze impact of single parameter.
        
        Args:
            parameter_name: Name of parameter to analyze
            
        Returns:
            Dictionary with analysis results
        """
        print(f"\n{'='*60}")
        print(f"ANALYZING PARAMETER: {parameter_name}")
        print(f"{'='*60}")
        
        # Load parameter ranges
        ranges_config = self.load_parameter_ranges()
        parameter_config = ranges_config["parameter_ranges"].get(parameter_name)
        experiment_settings = ranges_config.get("experiment_settings", {})
        
        if not parameter_config:
            raise ValueError(f"Parameter '{parameter_name}' not found in ranges configuration")
        
        values = parameter_config["values"]
        print(f"Testing values: {values}")
        
        # Results storage
        parameter_results = {
            "parameter_name": parameter_name,
            "parameter_config": parameter_config,
            "evaluation_mode": self.evaluation_mode,
            "timestamp": self.timestamp,
            "experiments": [],
            "summary": {}
        }
        
        # Run experiments for each parameter value
        for i, value in enumerate(values):
            print(f"\n{'-'*40}")
            print(f"Experiment {i+1}/{len(values)}: {parameter_name} = {value}")
            print(f"{'-'*40}")
            
            try:
                # Generate configuration
                configs = list(self.config_manager.generate_parameter_configs(parameter_name, [value], experiment_settings))
                config = configs[0]  # Single config for this value
                
                print(f"Training with config: {config['model_name']}")
                
                # Create temporary config file with the modified config
                import tempfile
                temp_config_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
                json.dump(config, temp_config_file)
                temp_config_file.close()
                
                # Train model with config file path
                training_results, model_path = self.trainer.train_model(temp_config_file.name)
                
                # Clean up temporary file
                import os
                os.unlink(temp_config_file.name)
                
                # Evaluate model
                evaluation_results = self.evaluator.evaluate_model(model_path, config)
                
                # Store experiment results
                experiment_result = {
                    "parameter_value": value,
                    "config": config,
                    "training_results": training_results,
                    "evaluation_results": evaluation_results,
                    "model_path": str(model_path),
                    "success": evaluation_results.get("success", False)
                }
                
                parameter_results["experiments"].append(experiment_result)
                
                print(f"Experiment completed successfully")
                if evaluation_results.get("success", False):
                    if self.evaluation_mode == "silicon_only":
                        metrics = evaluation_results.get("metrics", {})
                        print(f"  Metrics: {metrics}")
                    else:
                        transfer_metrics = evaluation_results.get("transfer_metrics", {})
                        print(f"  Transfer metrics: {transfer_metrics}")
                
            except Exception as e:
                print(f"Experiment failed: {e}")
                experiment_result = {
                    "parameter_value": value,
                    "config": config if 'config' in locals() else {},
                    "training_results": {},
                    "evaluation_results": {"success": False, "error": str(e)},
                    "model_path": "",
                    "success": False
                }
                parameter_results["experiments"].append(experiment_result)
        
        # Generate summary
        parameter_results["summary"] = self._generate_parameter_summary(parameter_results)
        
        # Generate report
        report_path = self.report_generator.generate_parameter_report(parameter_results)
        parameter_results["report_path"] = str(report_path)
        
        print(f"\nParameter analysis completed: {parameter_name}")
        print(f"Report saved: {report_path}")
        
        return parameter_results
    
    def analyze_multiple_parameters(self, parameter_names: List[str]) -> Dict[str, Any]:
        """
        Analyze multiple parameters.
        
        Args:
            parameter_names: List of parameter names to analyze
            
        Returns:
            Dictionary with combined analysis results
        """
        print(f"\n{'='*80}")
        print(f"MULTI-PARAMETER HYPERPARAMETER ANALYSIS")
        print(f"Parameters: {', '.join(parameter_names)}")
        print(f"Mode: {self.evaluation_mode}")
        print(f"{'='*80}")
        
        combined_results = {
            "analysis_type": "multi_parameter",
            "evaluation_mode": self.evaluation_mode,
            "timestamp": self.timestamp,
            "parameters": parameter_names,
            "parameter_results": {},
            "summary": {}
        }
        
        # Analyze each parameter
        for param_name in parameter_names:
            try:
                param_results = self.analyze_parameter(param_name)
                combined_results["parameter_results"][param_name] = param_results
            except Exception as e:
                print(f"Failed to analyze parameter {param_name}: {e}")
                combined_results["parameter_results"][param_name] = {
                    "error": str(e),
                    "success": False
                }
        
        # Generate combined summary
        combined_results["summary"] = self._generate_combined_summary(combined_results)
        
        # Generate combined report
        combined_report_path = self.report_generator.generate_combined_report(combined_results)
        combined_results["combined_report_path"] = str(combined_report_path)
        
        # Save experiment metadata
        self._save_experiment_metadata(combined_results)
        
        print(f"\nMulti-parameter analysis completed")
        print(f"Combined report: {combined_report_path}")
        print(f"Experiment directory: {self.experiment_dir}")
        
        return combined_results
    
    def _generate_parameter_summary(self, parameter_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics for parameter analysis"""
        experiments = parameter_results["experiments"]
        successful_experiments = [exp for exp in experiments if exp.get("success", False)]
        
        if not successful_experiments:
            return {"status": "no_successful_experiments"}
        
        summary = {
            "total_experiments": len(experiments),
            "successful_experiments": len(successful_experiments),
            "parameter_name": parameter_results["parameter_name"]
        }
        
        # Extract metrics based on evaluation mode
        if self.evaluation_mode == "silicon_only":
            metrics_data = []
            for exp in successful_experiments:
                metrics = exp.get("evaluation_results", {}).get("metrics", {})
                if metrics:
                    metrics["parameter_value"] = exp["parameter_value"]
                    metrics_data.append(metrics)
            
            if metrics_data:
                # Find best performing configuration using ranking system
                # Create ranking data similar to report_generator
                import pandas as pd
                
                df_data = []
                for exp in successful_experiments:
                    metrics = exp.get("evaluation_results", {}).get("metrics", {})
                    if metrics:
                        data_row = {"parameter_value": exp["parameter_value"]}
                        data_row.update(metrics)
                        df_data.append(data_row)
                
                if df_data:
                    df = pd.DataFrame(df_data)
                    
                    # Define metrics and their ranking direction
                    metrics_config = {
                        "rmse": "asc",              # Lower is better
                        "mae": "asc",               # Lower is better
                        "mape": "asc",              # Lower is better
                        "peak_error": "asc",        # Lower is better
                        "r2": "desc",               # Higher is better
                        "directional_accuracy": "desc"  # Higher is better
                    }
                    
                    # Calculate total points for each configuration
                    best_exp = None
                    best_total_points = float('inf')
                    
                    for _, row in df.iterrows():
                        total_points = 0
                        for metric, direction in metrics_config.items():
                            if metric in df.columns:
                                if direction == "asc":
                                    rank = df[metric].rank(method='min', ascending=True)[row.name]
                                else:
                                    rank = df[metric].rank(method='min', ascending=False)[row.name]
                                total_points += int(rank)
                        
                        if total_points < best_total_points:
                            best_total_points = total_points
                            best_exp = row
                    
                    if best_exp is not None:
                        # Find the corresponding experiment
                        best_exp_obj = next(exp for exp in successful_experiments 
                                          if exp["parameter_value"] == best_exp["parameter_value"])
                        
                        summary["best_configuration"] = {
                            "parameter_value": best_exp_obj["parameter_value"],
                            "metrics": best_exp_obj.get("evaluation_results", {}).get("metrics", {}),
                            "total_points": best_total_points
                        }
        
        return summary
    
    def _generate_combined_summary(self, combined_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary for multi-parameter analysis"""
        parameter_results = combined_results["parameter_results"]
        
        summary = {
            "total_parameters_analyzed": len(parameter_results),
            "successful_parameters": len([p for p in parameter_results.values() if not p.get("error")]),
            "evaluation_mode": self.evaluation_mode,
            "timestamp": combined_results["timestamp"]
        }
        
        return summary
    
    def _save_experiment_metadata(self, results: Dict[str, Any]):
        """Save experiment metadata"""
        metadata_path = self.experiment_dir / "experiment_metadata.json"
        
        # Create serializable metadata
        metadata = {
            "experiment_type": "hyperparameter_analysis",
            "evaluation_mode": self.evaluation_mode,
            "timestamp": self.timestamp,
            "parameters_analyzed": results.get("parameters", []),
            "total_experiments": sum(
                len(param_data.get("experiments", [])) 
                for param_data in results.get("parameter_results", {}).values()
                if isinstance(param_data, dict)
            ),
            "experiment_directory": str(self.experiment_dir),
            "reports_directory": str(self.reports_dir),
            "models_directory": str(self.trained_models_dir)
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"Experiment metadata saved: {metadata_path}")

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Run hyperparameter analysis experiments")
    parser.add_argument("--mode", default="silicon_only", choices=["silicon_only", "cross_technology"],
                       help="Evaluation mode")
    parser.add_argument("--parameters", required=True,
                       help="Parameters to analyze (comma-separated or 'all')")
    parser.add_argument("--config", default="baseline_lstm_model_config.json",
                       help="Base configuration file (in configs/ directory)")
    parser.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"],
                       help="Training device")
    
    args = parser.parse_args()
    
    # Parse parameters
    if args.parameters.lower() == "all":
        # Load all available parameters
        ranges_path = Path(__file__).parent / "configs" / "parameter_ranges.json"
        with open(ranges_path, 'r') as f:
            ranges_config = json.load(f)
        parameter_names = list(ranges_config["parameter_ranges"].keys())
    else:
        parameter_names = [p.strip() for p in args.parameters.split(",")]
    
    print(f"Starting hyperparameter analysis...")
    print(f"Mode: {args.mode}")
    print(f"Parameters: {parameter_names}")
    print(f"Device: {args.device}")
    
    try:
        # Initialize analysis
        analysis = HyperparameterAnalysis(args.mode, args.config)
        
        # Run analysis
        if len(parameter_names) == 1:
            results = analysis.analyze_parameter(parameter_names[0])
        else:
            results = analysis.analyze_multiple_parameters(parameter_names)
        
        print("\nHyperparameter analysis completed successfully!")
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # Direct configuration for easy testing
    # You can modify these parameters as needed
    
    # Analysis configuration
    MODE = "silicon_only"  # Options: "silicon_only", "cross_technology"
    PARAMETERS = "sequence_length"  # Options: "batch_size", "learning_rate", "epochs", "num_layers", "hidden_size", "sequence_length", "all"
    CONFIG = "baseline_lstm_model_config.json"  # Base configuration file
    DEVICE = "cuda"  # Options: "auto", "cpu", "cuda"
    
    print("=" * 60)
    print("HYPERPARAMETER ANALYSIS - DIRECT CONFIGURATION")
    print("=" * 60)
    print(f"Mode: {MODE}")
    print(f"Parameters: {PARAMETERS}")
    print(f"Config: {CONFIG}")
    print(f"Device: {DEVICE}")
    print("=" * 60)
    
    try:
        # Parse parameters
        if PARAMETERS.lower() == "all":
            # Load all available parameters
            ranges_path = Path(__file__).parent / "configs" / "parameter_ranges.json"
            with open(ranges_path, 'r') as f:
                ranges_config = json.load(f)
            parameter_names = list(ranges_config["parameter_ranges"].keys())
        else:
            parameter_names = [p.strip() for p in PARAMETERS.split(",")]
        
        print(f"Starting hyperparameter analysis...")
        print(f"Mode: {MODE}")
        print(f"Parameters: {parameter_names}")
        print(f"Device: {DEVICE}")
        
        # Initialize analysis
        analysis = HyperparameterAnalysis(MODE, CONFIG)
        
        # Run analysis
        if len(parameter_names) == 1:
            results = analysis.analyze_parameter(parameter_names[0])
        else:
            results = analysis.analyze_multiple_parameters(parameter_names)
        
        print("\nHyperparameter analysis completed successfully!")
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Uncomment the line below to also run the command line interface
    # main()
