import json
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any, Sequence
import warnings
from datetime import datetime
import os
import sys

# Set style for better plots
try:
    plt.style.use('seaborn-v0_8')
except:
    plt.style.use('seaborn')
sns.set_palette("husl")

class ModelAnalyzer:
    """
    Class for analyzing LSTM models.
    Supports both single model analysis and comparisons of multiple models.
    """
    
    def __init__(self, model_paths: Union[str, Path, Sequence[Union[str, Path]]], 
                 output_dir: Optional[Union[str, Path]] = None):
        """
        Initializes the ModelAnalyzer.
        
        Args:
            model_paths: Path to a model or list of model paths
            output_dir: Output directory for plots (optional)
        """
        # Convert to list for uniform handling
        if isinstance(model_paths, (str, Path)):
            self.model_paths = [Path(model_paths)]
        else:
            self.model_paths = [Path(p) for p in model_paths]
        
        # Validate paths
        for path in self.model_paths:
            if not path.exists():
                raise FileNotFoundError(f"Model path does not exist: {path}")
        
        # Determine if single model or multi-model analysis
        self.is_single_model = len(self.model_paths) == 1
        
        # Set output directory based on analysis type
        if output_dir is None:
            if self.is_single_model:
                # Single model: Use same name as model folder
                model_name = self.model_paths[0].name
                self.output_dir = Path(f"results/model_analysis/{model_name}")
            else:
                # Multi-model: Use combined_analysis with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.output_dir = Path(f"results/model_analysis/combined_analysis_{timestamp}")
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create comparison file for multi-model analysis
        if not self.is_single_model:
            self._create_comparison_info_file()
        
        # Load model information
        self.models_data = self._load_models_data()
    
    def _create_comparison_info_file(self) -> None:
        """Creates a file with information about the compared models."""
        comparison_info = []
        comparison_info.append("=" * 80)
        comparison_info.append("COMPARED MODELS")
        comparison_info.append("=" * 80)
        comparison_info.append(f"Created at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        comparison_info.append(f"Number of models: {len(self.model_paths)}")
        comparison_info.append("")
        comparison_info.append("MODEL PATHS:")
        comparison_info.append("-" * 40)
        
        for i, model_path in enumerate(self.model_paths, 1):
            comparison_info.append(f"{i}. {model_path.name}")
            comparison_info.append(f"   Full path: {model_path}")
            comparison_info.append("")
        
        comparison_info.append("ANALYSIS FOLDER:")
        comparison_info.append("-" * 40)
        comparison_info.append(f"Output Directory: {self.output_dir}")
        comparison_info.append("")
        comparison_info.append("=" * 80)
        
        # Save comparison information
        comparison_file = self.output_dir / "compared_models.txt"
        with open(comparison_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(comparison_info))
        
        print(f"Comparison info saved to: {comparison_file}")
        
    def _load_models_data(self) -> Dict[str, Dict[str, Any]]:
        """Loads all model information."""
        models_data = {}
        
        print(f"[DEBUG] Loading data for {len(self.model_paths)} models...")
        
        for model_path in self.model_paths:
            model_name = model_path.name
            print(f"[DEBUG] Processing model: {model_name} at {model_path}")
            
            # Load Training History
            history_file = model_path / f"training_history_{model_name}.json"
            if history_file.exists():
                with open(history_file, 'r') as f:
                    training_history = json.load(f)
                print(f"[DEBUG] Loaded training history for {model_name}: {len(training_history.get('train_loss', []))} epochs")
            else:
                print(f"[WARNING] No training history file found for {model_name}: {history_file}")
                training_history = {"train_loss": [], "val_loss": []}
            
            # Load Model Config
            config_file = model_path / f"model_config_{model_name}.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    model_config = json.load(f)
                print(f"[DEBUG] Loaded model config for {model_name}")
            else:
                print(f"[WARNING] No model config file found for {model_name}: {config_file}")
                model_config = {}
            
            # Load Training Summary
            summary_file = model_path / f"training_summary_{model_name}.txt"
            training_summary = ""
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    training_summary = f.read()
                print(f"[DEBUG] Loaded training summary for {model_name}")
            else:
                print(f"[WARNING] No training summary file found for {model_name}: {summary_file}")
            
            # Validate training history
            if not training_history.get('train_loss') or len(training_history['train_loss']) == 0:
                print(f"[ERROR] No training loss data found for {model_name}")
                continue
            
            models_data[model_name] = {
                'path': model_path,
                'training_history': training_history,
                'model_config': model_config,
                'training_summary': training_summary
            }
            
            print(f"[SUCCESS] Successfully loaded data for {model_name}")
        
        print(f"[INFO] Successfully loaded {len(models_data)} models with valid data")
        return models_data
    
    def plot_training_curves(self, save_plot: bool = True, show_plot: bool = True) -> Figure:
        """
        Creates plots of training and validation curves.
        
        Args:
            save_plot: Whether to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            matplotlib Figure object
        """
        if not self.models_data:
            print("[ERROR] No valid model data available for plotting")
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Use a different color map that's more reliable
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        valid_models = 0
        
        for idx, (model_name, data) in enumerate(self.models_data.items()):
            history = data['training_history']
            color = colors[idx % len(colors)]
            
            # Check if we have valid training data
            if not history.get('train_loss') or len(history['train_loss']) == 0:
                print(f"[WARNING] No training loss data for {model_name}, skipping...")
                continue
            
            epochs = range(1, len(history['train_loss']) + 1)
            valid_models += 1
            
            # Training Loss
            axes[0].plot(epochs, history['train_loss'], 
                        label=f'{model_name} (Train)', 
                        color=color, linewidth=2, alpha=0.8)
            
            # Validation Loss (falls vorhanden)
            if history.get('val_loss') and len(history['val_loss']) > 0:
                axes[0].plot(epochs, history['val_loss'], 
                           label=f'{model_name} (Val)', 
                           color=color, linewidth=2, linestyle='--', alpha=0.8)
        
        if valid_models == 0:
            print("[ERROR] No valid models with training data found")
            plt.close(fig)
            return None
        
        axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epochs', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        if valid_models > 1:
            axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0].grid(True, alpha=0.3)
        
        # Loss Ratio Plot (Train/Val)
        for idx, (model_name, data) in enumerate(self.models_data.items()):
            history = data['training_history']
            color = colors[idx % len(colors)]
            
            if history.get('val_loss') and len(history['val_loss']) > 0 and len(history['train_loss']) > 0:
                epochs = range(1, len(history['train_loss']) + 1)
                loss_ratio = [t/v if v > 0 else 0 for t, v in zip(history['train_loss'], history['val_loss'])]
                
                axes[1].plot(epochs, loss_ratio, 
                           label=f'{model_name}', 
                           color=color, linewidth=2, marker='o', markersize=4)
        
        axes[1].set_title('Training/Validation Loss Ratio', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epochs', fontsize=12)
        axes[1].set_ylabel('Train Loss / Val Loss', fontsize=12)
        if valid_models > 1:
            axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Ratio = 1')
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = self.output_dir / 'training_curves.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"[SUCCESS] Training curves saved to: {plot_path}")
        
        if show_plot:
            plt.show()
        
        return fig
    
    def plot_model_architecture_comparison(self, save_plot: bool = True, show_plot: bool = True) -> plt.Figure:
        """
        Creates a visual comparison of model architectures.
        
        Args:
            save_plot: Whether to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Collect data for plots
        model_names = []
        hidden_sizes = []
        num_layers = []
        dropout_rates = []
        sequence_lengths = []
        
        for model_name, data in self.models_data.items():
            config = data['model_config']
            model_names.append(model_name)
            hidden_sizes.append(config.get('hidden_size', 0))
            num_layers.append(config.get('num_layers', 0))
            dropout_rates.append(config.get('dropout', 0))
            sequence_lengths.append(config.get('sequence_length', 0))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # Hidden Size Comparison
        axes[0, 0].bar(model_names, hidden_sizes, color=colors[:len(model_names)], alpha=0.7)
        axes[0, 0].set_title('Hidden Size Comparison', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Hidden Size', fontsize=12)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Number of Layers Comparison
        axes[0, 1].bar(model_names, num_layers, color=colors[:len(model_names)], alpha=0.7)
        axes[0, 1].set_title('Number of LSTM Layers', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('Number of Layers', fontsize=12)
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Dropout Rate Comparison
        axes[1, 0].bar(model_names, dropout_rates, color=colors[:len(model_names)], alpha=0.7)
        axes[1, 0].set_title('Dropout Rate', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('Dropout Rate', fontsize=12)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Sequence Length Comparison
        axes[1, 1].bar(model_names, sequence_lengths, color=colors[:len(model_names)], alpha=0.7)
        axes[1, 1].set_title('Sequence Length', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('Sequence Length', fontsize=12)
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = self.output_dir / 'model_architecture_comparison.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Model architecture comparison saved to: {plot_path}")
        
        if show_plot:
            plt.show()
        
        return fig
    
    def plot_training_settings_comparison(self, save_plot: bool = True, show_plot: bool = True) -> plt.Figure:
        """
        Creates a comparison of training parameters.
        
        Args:
            save_plot: Whether to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Collect data
        model_names = []
        learning_rates = []
        batch_sizes = []
        epochs = []
        validation_splits = []
        
        for model_name, data in self.models_data.items():
            config = data['model_config']
            training_settings = config.get('training_settings', {})
            
            model_names.append(model_name)
            learning_rates.append(training_settings.get('learning_rate', 0))
            batch_sizes.append(training_settings.get('batch_size', 0))
            epochs.append(training_settings.get('epochs', 0))
            validation_splits.append(training_settings.get('validation_split', 0))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # Learning Rate Comparison
        axes[0, 0].bar(model_names, learning_rates, color=colors[:len(model_names)], alpha=0.7)
        axes[0, 0].set_title('Learning Rate', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Learning Rate', fontsize=12)
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].set_yscale('log')
        
        # Batch Size Comparison
        axes[0, 1].bar(model_names, batch_sizes, color=colors[:len(model_names)], alpha=0.7)
        axes[0, 1].set_title('Batch Size', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('Batch Size', fontsize=12)
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Epochs Comparison
        axes[1, 0].bar(model_names, epochs, color=colors[:len(model_names)], alpha=0.7)
        axes[1, 0].set_title('Number of Epochs', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('Epochs', fontsize=12)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Validation Split Comparison
        axes[1, 1].bar(model_names, validation_splits, color=colors[:len(model_names)], alpha=0.7)
        axes[1, 1].set_title('Validation Split', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('Validation Split', fontsize=12)
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = self.output_dir / 'training_settings_comparison.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Training settings comparison saved to: {plot_path}")
        
        if show_plot:
            plt.show()
        
        return fig
    
    def plot_convergence_analysis(self, save_plot: bool = True, show_plot: bool = True) -> plt.Figure:
        """
        Analyzes the convergence of models.
        
        Args:
            save_plot: Whether to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        for idx, (model_name, data) in enumerate(self.models_data.items()):
            history = data['training_history']
            color = colors[idx]
            
            epochs = range(1, len(history['train_loss']) + 1)
            
            # Loss Gradient (change rate)
            if len(history['train_loss']) > 1:
                loss_gradient = np.diff(history['train_loss'])
                axes[0, 0].plot(epochs[1:], loss_gradient, 
                               label=f'{model_name}', 
                               color=color, linewidth=2, marker='o', markersize=4)
            
            # Loss Log Scale
            axes[0, 1].semilogy(epochs, history['train_loss'], 
                               label=f'{model_name}', 
                               color=color, linewidth=2, marker='o', markersize=4)
            
            # Final Loss Values
            final_train_loss = history['train_loss'][-1] if history['train_loss'] else 0
            final_val_loss = history['val_loss'][-1] if history.get('val_loss') else 0
            
            axes[1, 0].bar(f'{model_name}\n(Train)', final_train_loss, 
                          color=color, alpha=0.7, width=0.35)
            if final_val_loss > 0:
                axes[1, 0].bar(f'{model_name}\n(Val)', final_val_loss, 
                              color=color, alpha=0.4, width=0.35)
            
            # Convergence Speed (Epochen bis 90% des finalen Loss)
            if len(history['train_loss']) > 1:
                final_loss = history['train_loss'][-1]
                target_loss = final_loss * 1.1  # 110% des finalen Loss
                
                convergence_epoch = None
                for i, loss in enumerate(history['train_loss']):
                    if loss <= target_loss:
                        convergence_epoch = i + 1
                        break
                
                if convergence_epoch:
                    axes[1, 1].bar(model_name, convergence_epoch, 
                                  color=colors[idx % len(colors)], alpha=0.7)
        
        axes[0, 0].set_title('Loss Gradient (Convergence Speed)', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epochs', fontsize=12)
        axes[0, 0].set_ylabel('Loss Change', fontsize=12)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        axes[0, 1].set_title('Training Loss (Log Scale)', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epochs', fontsize=12)
        axes[0, 1].set_ylabel('Loss (log)', fontsize=12)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_title('Final Loss Values', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('Final Loss', fontsize=12)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        axes[1, 1].set_title('Convergence Speed', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('Epochs until 110% of final loss', fontsize=12)
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = self.output_dir / 'convergence_analysis.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Convergence analysis saved to: {plot_path}")
        
        if show_plot:
            plt.show()
        
        return fig
    
    def create_summary_report(self, save_report: bool = True) -> str:
        """
        Creates a detailed summary report.
        
        Args:
            save_report: Whether to save the report
            
        Returns:
            Report as string
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("LSTM MODEL ANALYSIS REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Created at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Number of analyzed models: {len(self.models_data)}")
        report_lines.append("")
        
        for model_name, data in self.models_data.items():
            report_lines.append(f"MODEL: {model_name}")
            report_lines.append("-" * 40)
            
            # Model Config
            config = data['model_config']
            report_lines.append("ARCHITECTURE:")
            report_lines.append(f"  - Hidden Size: {config.get('hidden_size', 'N/A')}")
            report_lines.append(f"  - LSTM Layers: {config.get('num_layers', 'N/A')}")
            report_lines.append(f"  - Dropout: {config.get('dropout', 'N/A')}")
            report_lines.append(f"  - Input Size: {config.get('input_size', 'N/A')}")
            report_lines.append(f"  - Output Size: {config.get('output_size', 'N/A')}")
            report_lines.append(f"  - Sequence Length: {config.get('sequence_length', 'N/A')}")
            
            # Training Settings
            training_settings = config.get('training_settings', {})
            report_lines.append("TRAINING PARAMETERS:")
            report_lines.append(f"  - Learning Rate: {training_settings.get('learning_rate', 'N/A')}")
            report_lines.append(f"  - Batch Size: {training_settings.get('batch_size', 'N/A')}")
            report_lines.append(f"  - Epochs: {training_settings.get('epochs', 'N/A')}")
            report_lines.append(f"  - Loss Function: {training_settings.get('loss_function', 'N/A')}")
            report_lines.append(f"  - Validation Split: {training_settings.get('validation_split', 'N/A')}")
            report_lines.append(f"  - Features: {training_settings.get('features', 'N/A')}")
            report_lines.append(f"  - Output Features: {training_settings.get('output', 'N/A')}")
            
            # Training Performance
            history = data['training_history']
            if history.get('train_loss') and len(history['train_loss']) > 0:
                report_lines.append("TRAINING PERFORMANCE:")
                report_lines.append(f"  - Initial Train Loss: {history['train_loss'][0]:.6f}")
                report_lines.append(f"  - Final Train Loss: {history['train_loss'][-1]:.6f}")
                report_lines.append(f"  - Loss Reduction: {((history['train_loss'][0] - history['train_loss'][-1]) / history['train_loss'][0] * 100):.2f}%")
                
                if history.get('val_loss') and len(history['val_loss']) > 0:
                    report_lines.append(f"  - Initial Val Loss: {history['val_loss'][0]:.6f}")
                    report_lines.append(f"  - Final Val Loss: {history['val_loss'][-1]:.6f}")
                    report_lines.append(f"  - Final Train/Val Ratio: {history['train_loss'][-1] / history['val_loss'][-1]:.3f}")
                    
                    # Overfitting Check
                    final_ratio = history['train_loss'][-1] / history['val_loss'][-1]
                    if final_ratio < 0.8:
                        report_lines.append("  - WARNING: Possible underfitting (Train Loss < Val Loss)")
                    elif final_ratio > 1.2:
                        report_lines.append("  - WARNING: Possible overfitting (Train Loss > Val Loss)")
                    else:
                        report_lines.append("  - Good balance between training and validation")
            else:
                report_lines.append("TRAINING PERFORMANCE:")
                report_lines.append("  - No training loss data available")
            
            report_lines.append("")
        
        # Model comparison
        if len(self.models_data) > 1:
            report_lines.append("MODEL COMPARISON:")
            report_lines.append("-" * 40)
            
            # Best model based on final loss
            train_losses = []
            for name, data in self.models_data.items():
                if data['training_history'].get('train_loss') and len(data['training_history']['train_loss']) > 0:
                    train_losses.append((name, data['training_history']['train_loss'][-1]))
            
            if train_losses:
                best_model = min(train_losses, key=lambda x: x[1])
                report_lines.append(f"Best Train Loss: {best_model[0]} ({best_model[1]:.6f})")
            
            # Best model based on validation loss
            val_losses = []
            for name, data in self.models_data.items():
                if data['training_history'].get('val_loss') and len(data['training_history']['val_loss']) > 0:
                    val_losses.append((name, data['training_history']['val_loss'][-1]))
            
            if val_losses:
                best_val_model = min(val_losses, key=lambda x: x[1])
                report_lines.append(f"Best Val Loss: {best_val_model[0]} ({best_val_model[1]:.6f})")
        
        report = "\n".join(report_lines)
        
        if save_report:
            report_path = self.output_dir / 'model_analysis_report.txt'
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"Analysis report saved to: {report_path}")
        
        return report
    
    def run_complete_analysis(self, save_all: bool = True, show_plots: bool = True) -> Dict[str, Any]:
        """
        Performs a complete analysis of all models.
        
        Args:
            save_all: Whether to save all plots and reports
            show_plots: Whether to display plots
            
        Returns:
            Dictionary with all analysis results
        """
        print("Starting complete model analysis...")
        
        results = {}
        
        # Training Curves
        print("Creating training curves...")
        results['training_curves'] = self.plot_training_curves(save_plot=save_all, show_plot=show_plots)
        
        # Model Architecture Comparison
        print("Creating model architecture comparison...")
        results['architecture_comparison'] = self.plot_model_architecture_comparison(save_plot=save_all, show_plot=show_plots)
        
        # Training Settings Comparison
        print("Creating training settings comparison...")
        results['settings_comparison'] = self.plot_training_settings_comparison(save_plot=save_all, show_plot=show_plots)
        
        # Convergence Analysis
        print("Creating convergence analysis...")
        results['convergence_analysis'] = self.plot_convergence_analysis(save_plot=save_all, show_plot=show_plots)
        
        # Summary Report
        print("Creating summary report...")
        results['summary_report'] = self.create_summary_report(save_report=save_all)
        
        print(f"Complete analysis finished. Results saved to: {self.output_dir}")
        
        return results


# Example usage
if __name__ == "__main__":
    # Example for single model analysis
    # analyzer = ModelAnalyzer("results/trained_models/lstm/20250706_2137_lstm")
    
    # Example for multi-model analysis
    model_paths = [
        "results/trained_models/lstm/20250706_2137_lstm",
        "results/trained_models/lstm/20250706_2134_lstm",
        "results/trained_models/lstm/20250706_2102_lstm"
    ]
    
    # Only use models that exist
    existing_paths = [p for p in model_paths if Path(p).exists()]
    
    if existing_paths:
        analyzer = ModelAnalyzer(existing_paths)
        results = analyzer.run_complete_analysis(save_all=True, show_plots=True)
        print("Analysis completed successfully!")
    else:
        print("No existing model paths found. Please check the paths.") 