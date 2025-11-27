"""
Analytical Package 4: Scatter Plot Region Analysis
==================================================

This package analyzes model performance across different power level regions
to assess transferability between module types. It divides the scatter plot
into power level regions and evaluates performance metrics for each region.

Focus: Transferability analysis through region-specific performance evaluation
to identify where Silicon vs. Perovskite models perform differently.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy import stats
import json
import os


class ScatterRegionAnalysis:
    """
    Analytical Package 4: Scatter Plot Region Analysis for PV Forecasting
    
    Analyzes model performance across different power level regions to assess
    transferability between module types.
    """
    
    def __init__(self, module_type: str):
        """
        Initialize Scatter Region Analysis for specific module type
        
        Args:
            module_type: 'silicon' or 'perovskite'
        """
        self.module_type = module_type
        self.results = {}
        
    def analyze_scatter_plot_regions(self, predictions: np.ndarray, 
                                   actuals: np.ndarray, 
                                   timestamps: pd.DatetimeIndex) -> Dict[str, Any]:
        """
        Analyze scatter plot regions to assess transferability between module types
        
        Args:
            predictions: Model predictions (n_samples, forecast_steps, n_features)
            actuals: Actual values (n_samples, forecast_steps, n_features)
            timestamps: Corresponding timestamps
            
        Returns:
            Dictionary with region analysis results
        """
        print(f"[INFO] Analyzing scatter plot regions for {self.module_type} modules...")
        
        # Flatten predictions and actuals
        predictions_flat = predictions.flatten()
        actuals_flat = actuals.flatten()
        
        # Define power level regions (based on normalized values 0-1.2)
        regions = {
            'very_low': (0.0, 0.2),
            'low': (0.2, 0.4), 
            'medium': (0.4, 0.6),
            'high': (0.6, 0.8),
            'very_high': (0.8, 1.0),
            'extreme': (1.0, 1.2)
        }
        
        region_results = {}
        
        for region_name, (min_val, max_val) in regions.items():
            # Find indices for this region
            region_mask = (actuals_flat >= min_val) & (actuals_flat < max_val)
            
            if np.sum(region_mask) > 10:  # Need minimum samples for meaningful analysis
                region_actuals = actuals_flat[region_mask]
                region_predictions = predictions_flat[region_mask]
                
                # Calculate metrics for this region
                mae = np.mean(np.abs(region_predictions - region_actuals))
                rmse = np.sqrt(np.mean((region_predictions - region_actuals)**2))
                r2 = self._calculate_r2(region_actuals, region_predictions)
                
                # Calculate bias (mean error)
                bias = np.mean(region_predictions - region_actuals)
                
                # Calculate percentage of points within different error thresholds
                abs_errors = np.abs(region_predictions - region_actuals)
                pct_within_5 = np.sum(abs_errors <= 0.05) / len(abs_errors) * 100
                pct_within_10 = np.sum(abs_errors <= 0.10) / len(abs_errors) * 100
                pct_within_20 = np.sum(abs_errors <= 0.20) / len(abs_errors) * 100
                
                # Additional metrics for transferability analysis
                correlation = np.corrcoef(region_actuals, region_predictions)[0, 1]
                mape = np.mean(np.abs(region_predictions - region_actuals) / (region_actuals + 1e-8)) * 100
                
                region_results[region_name] = {
                    'power_range': f"{min_val:.1f}-{max_val:.1f}",
                    'sample_count': int(np.sum(region_mask)),
                    'mae': float(mae),
                    'rmse': float(rmse),
                    'r2': float(r2),
                    'bias': float(bias),
                    'correlation': float(correlation),
                    'mape': float(mape),
                    'pct_within_5pct': float(pct_within_5),
                    'pct_within_10pct': float(pct_within_10),
                    'pct_within_20pct': float(pct_within_20),
                    'mean_actual': float(np.mean(region_actuals)),
                    'mean_predicted': float(np.mean(region_predictions)),
                    'std_actual': float(np.std(region_actuals)),
                    'std_predicted': float(np.std(region_predictions))
                }
        
        # Overall analysis
        overall_mae = np.mean(np.abs(predictions_flat - actuals_flat))
        overall_rmse = np.sqrt(np.mean((predictions_flat - actuals_flat)**2))
        overall_r2 = self._calculate_r2(actuals_flat, predictions_flat)
        overall_bias = np.mean(predictions_flat - actuals_flat)
        overall_correlation = np.corrcoef(actuals_flat, predictions_flat)[0, 1]
        overall_mape = np.mean(np.abs(predictions_flat - actuals_flat) / (actuals_flat + 1e-8)) * 100
        
        # Calculate overall error distribution
        abs_errors = np.abs(predictions_flat - actuals_flat)
        overall_pct_within_5 = np.sum(abs_errors <= 0.05) / len(abs_errors) * 100
        overall_pct_within_10 = np.sum(abs_errors <= 0.10) / len(abs_errors) * 100
        overall_pct_within_20 = np.sum(abs_errors <= 0.20) / len(abs_errors) * 100
        
        results = {
            'module_type': self.module_type,
            'regions': region_results,
            'overall': {
                'mae': float(overall_mae),
                'rmse': float(overall_rmse),
                'r2': float(overall_r2),
                'bias': float(overall_bias),
                'correlation': float(overall_correlation),
                'mape': float(overall_mape),
                'pct_within_5pct': float(overall_pct_within_5),
                'pct_within_10pct': float(overall_pct_within_10),
                'pct_within_20pct': float(overall_pct_within_20),
                'total_samples': len(actuals_flat),
                'mean_actual': float(np.mean(actuals_flat)),
                'mean_predicted': float(np.mean(predictions_flat)),
                'std_actual': float(np.std(actuals_flat)),
                'std_predicted': float(np.std(predictions_flat))
            }
        }
        
        # Transferability insights
        results['transferability_insights'] = self._analyze_transferability_patterns(region_results)
        
        print(f"[INFO] Scatter region analysis completed for {self.module_type}:")
        print(f"  - Regions analyzed: {len(region_results)}")
        print(f"  - Total data points: {len(actuals_flat)}")
        print(f"  - Overall R²: {overall_r2:.4f}")
        print(f"  - Overall MAE: {overall_mae:.4f}")
        
        return results
    
    def _calculate_r2(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate R² score"""
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    def _analyze_transferability_patterns(self, region_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze patterns that indicate transferability characteristics"""
        
        insights = {
            'best_performing_region': None,
            'worst_performing_region': None,
            'most_stable_region': None,
            'most_biased_region': None,
            'performance_consistency': None,
            'bias_patterns': {}
        }
        
        if not region_results:
            return insights
        
        # Find best and worst performing regions by R²
        r2_scores = {region: data['r2'] for region, data in region_results.items()}
        if r2_scores:
            insights['best_performing_region'] = max(r2_scores.items(), key=lambda x: x[1])[0]
            insights['worst_performing_region'] = min(r2_scores.items(), key=lambda x: x[1])[0]
        
        # Find most stable region (lowest MAE)
        mae_scores = {region: data['mae'] for region, data in region_results.items()}
        if mae_scores:
            insights['most_stable_region'] = min(mae_scores.items(), key=lambda x: x[1])[0]
        
        # Find most biased region (highest absolute bias)
        bias_scores = {region: abs(data['bias']) for region, data in region_results.items()}
        if bias_scores:
            insights['most_biased_region'] = max(bias_scores.items(), key=lambda x: x[1])[0]
        
        # Analyze bias patterns
        for region, data in region_results.items():
            bias = data['bias']
            if bias > 0.05:
                insights['bias_patterns'][region] = 'overprediction'
            elif bias < -0.05:
                insights['bias_patterns'][region] = 'underprediction'
            else:
                insights['bias_patterns'][region] = 'unbiased'
        
        # Calculate performance consistency (coefficient of variation of R² scores)
        if len(r2_scores) > 1:
            r2_values = list(r2_scores.values())
            r2_cv = np.std(r2_values) / (np.mean(r2_values) + 1e-8)
            if r2_cv < 0.1:
                insights['performance_consistency'] = 'high'
            elif r2_cv < 0.3:
                insights['performance_consistency'] = 'medium'
            else:
                insights['performance_consistency'] = 'low'
        
        return insights
    
    def create_scatter_region_visualizations(self, region_results: Dict[str, Any], 
                                           save_path: str = None) -> Dict[str, plt.Figure]:
        """
        Create visualizations for scatter region analysis
        
        Args:
            region_results: Results from analyze_scatter_plot_regions
            save_path: Optional path to save figures
            
        Returns:
            Dictionary of matplotlib figures
        """
        figures = {}
        
        if not region_results or 'regions' not in region_results:
            print("[WARN] No scatter region data available for visualization")
            return figures
        
        regions = region_results['regions']
        
        # 1. Performance Metrics by Region
        fig1, ((ax1a, ax1b), (ax1c, ax1d)) = plt.subplots(2, 2, figsize=(15, 12))
        
        region_names = list(regions.keys())
        region_labels = [name.replace('_', ' ').title() for name in region_names]
        
        # R² scores
        r2_scores = [regions[name]['r2'] for name in region_names]
        ax1a.bar(region_labels, r2_scores, alpha=0.7, color='skyblue')
        ax1a.set_xlabel('Power Region')
        ax1a.set_ylabel('R² Score')
        ax1a.set_title(f'R² Score by Power Region - {self.module_type.title()} Modules')
        ax1a.tick_params(axis='x', rotation=45)
        ax1a.grid(True, alpha=0.3)
        
        # MAE scores
        mae_scores = [regions[name]['mae'] for name in region_names]
        ax1b.bar(region_labels, mae_scores, alpha=0.7, color='lightcoral')
        ax1b.set_xlabel('Power Region')
        ax1b.set_ylabel('MAE')
        ax1b.set_title(f'MAE by Power Region - {self.module_type.title()} Modules')
        ax1b.tick_params(axis='x', rotation=45)
        ax1b.grid(True, alpha=0.3)
        
        # Bias
        bias_scores = [regions[name]['bias'] for name in region_names]
        colors = ['red' if b < 0 else 'blue' for b in bias_scores]
        ax1c.bar(region_labels, bias_scores, alpha=0.7, color=colors)
        ax1c.set_xlabel('Power Region')
        ax1c.set_ylabel('Bias')
        ax1c.set_title(f'Prediction Bias by Power Region - {self.module_type.title()} Modules')
        ax1c.tick_params(axis='x', rotation=45)
        ax1c.grid(True, alpha=0.3)
        ax1c.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Within 10% accuracy
        within_10_scores = [regions[name]['pct_within_10pct'] for name in region_names]
        ax1d.bar(region_labels, within_10_scores, alpha=0.7, color='lightgreen')
        ax1d.set_xlabel('Power Region')
        ax1d.set_ylabel('Within 10% (%)')
        ax1d.set_title(f'Predictions Within 10% Error - {self.module_type.title()} Modules')
        ax1d.tick_params(axis='x', rotation=45)
        ax1d.grid(True, alpha=0.3)
        
        plt.tight_layout()
        figures['region_performance'] = fig1
        
        # 2. Sample Distribution by Region
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        
        sample_counts = [regions[name]['sample_count'] for name in region_names]
        percentages = [count / sum(sample_counts) * 100 for count in sample_counts]
        
        bars = ax2.bar(region_labels, percentages, alpha=0.7, color='orange')
        ax2.set_xlabel('Power Region')
        ax2.set_ylabel('Percentage of Samples (%)')
        ax2.set_title(f'Sample Distribution by Power Region - {self.module_type.title()} Modules')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add count labels on bars
        for bar, count in zip(bars, sample_counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{count:,}', ha='center', va='bottom')
        
        figures['sample_distribution'] = fig2
        
        # Save figures if path provided
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            for name, fig in figures.items():
                fig.savefig(f"{save_path}/scatter_region_analysis_{name}_{self.module_type}.png", 
                           dpi=300, bbox_inches='tight')
        
        return figures
    
    def save_results_to_file(self, results: Dict[str, Any], output_dir: str) -> str:
        """
        Save analysis results to JSON file for later use
        
        Args:
            results: Analysis results
            output_dir: Output directory
            
        Returns:
            Path to saved file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"scatter_region_analysis_{self.module_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"[INFO] Scatter region analysis results saved to: {filepath}")
        return filepath


def analyze_scatter_plot_regions(predictions: np.ndarray, 
                                actuals: np.ndarray, 
                                timestamps: pd.DatetimeIndex,
                                module_type: str = 'silicon') -> Dict[str, Any]:
    """
    Convenience function to analyze scatter plot regions
    
    Args:
        predictions: Model predictions
        actuals: Actual values  
        timestamps: Corresponding timestamps
        module_type: 'silicon' or 'perovskite'
        
    Returns:
        Scatter region analysis results
    """
    analyzer = ScatterRegionAnalysis(module_type)
    return analyzer.analyze_scatter_plot_regions(predictions, actuals, timestamps)


if __name__ == "__main__":
    # Example usage
    print("Analytical Package 4: Scatter Plot Region Analysis")
    print("This package analyzes model performance across different power level regions")
    print("to assess transferability between module types.")
