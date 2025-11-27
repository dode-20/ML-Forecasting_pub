"""
Analytical Package 1: Variability Analysis
==========================================

This package analyzes the prediction quality of LSTM models for different
variability characteristics of Silicon vs. Perovskite modules.

Focus: How well can the LSTM model predict the variability patterns
of different module types, not just the general module differences.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta


class VariabilityAnalysis:
    """
    Analytical Package 1: Variability Analysis for PV Forecasting
    
    Analyzes how well LSTM models can predict variability characteristics
    of Silicon vs. Perovskite modules.
    """
    
    def __init__(self, module_type: str):
        """
        Initialize Variability Analysis for specific module type
        
        Args:
            module_type: 'silicon' or 'perovskite'
        """
        self.module_type = module_type
        self.results = {}
        
    def analyze_cv_prediction_quality(self, predictions: np.ndarray, 
                                    actuals: np.ndarray, 
                                    timestamps: pd.DatetimeIndex) -> Dict[str, Any]:
        """
        Analyze how well the model predicts daily coefficient of variation (CV)
        
        Args:
            predictions: Model predictions (n_samples, forecast_steps, n_features)
            actuals: Actual values (n_samples, forecast_steps, n_features)
            timestamps: Corresponding timestamps
            
        Returns:
            Dictionary with CV prediction quality analysis
        """
        print(f"[INFO] Analyzing CV prediction quality for {self.module_type} modules...")
        
        # Flatten predictions and actuals for daily analysis
        if len(predictions.shape) == 3:
            predictions_flat = predictions.reshape(-1, predictions.shape[-1])
            actuals_flat = actuals.reshape(-1, actuals.shape[-1])
        else:
            predictions_flat = predictions
            actuals_flat = actuals
            
        # Ensure timestamps match flattened data
        if len(timestamps) != len(predictions_flat):
            # For multi-step predictions, we need to expand timestamps
            timestamps_flat = self._expand_timestamps_for_multistep(timestamps, predictions.shape[1])
        else:
            timestamps_flat = timestamps
            
        # Convert to DataFrame for easier daily grouping
        df = pd.DataFrame({
            'timestamp': timestamps_flat,
            'predicted': predictions_flat.flatten(),
            'actual': actuals_flat.flatten()
        })
        
        # Add date column for daily grouping
        df['date'] = df['timestamp'].dt.date
        
        # Calculate daily CV for predictions and actuals
        daily_cv_data = []
        
        for date in df['date'].unique():
            day_data = df[df['date'] == date]
            
            if len(day_data) < 2:  # Need at least 2 points for CV calculation
                continue
                
            # Calculate CV for this day
            pred_cv = self._calculate_cv(day_data['predicted'].values)
            actual_cv = self._calculate_cv(day_data['actual'].values)
            
            # Calculate CV prediction error
            cv_error = abs(pred_cv - actual_cv)
            cv_error_relative = cv_error / actual_cv if actual_cv > 0 else 0
            
            daily_cv_data.append({
                'date': date,
                'predicted_cv': pred_cv,
                'actual_cv': actual_cv,
                'cv_error_absolute': cv_error,
                'cv_error_relative': cv_error_relative,
                'n_points': len(day_data)
            })
        
        daily_cv_df = pd.DataFrame(daily_cv_data)
        
        if len(daily_cv_df) == 0:
            print(f"[WARN] No valid daily data for CV analysis")
            return {}
        
        # Calculate summary statistics
        summary_stats = {
            'mean_cv_error_absolute': daily_cv_df['cv_error_absolute'].mean(),
            'std_cv_error_absolute': daily_cv_df['cv_error_absolute'].std(),
            'mean_cv_error_relative': daily_cv_df['cv_error_relative'].mean(),
            'std_cv_error_relative': daily_cv_df['cv_error_relative'].std(),
            'mean_predicted_cv': daily_cv_df['predicted_cv'].mean(),
            'mean_actual_cv': daily_cv_df['actual_cv'].mean(),
            'cv_correlation': daily_cv_df['predicted_cv'].corr(daily_cv_df['actual_cv']),
            'n_days': len(daily_cv_df)
        }
        
        # Identify problematic days (high CV prediction errors)
        threshold_high_error = daily_cv_df['cv_error_relative'].quantile(0.75)
        problematic_days = daily_cv_df[daily_cv_df['cv_error_relative'] > threshold_high_error]
        
        # Weather condition analysis (if available)
        weather_analysis = self._analyze_cv_weather_dependency(daily_cv_df, df)
        
        results = {
            'summary_stats': summary_stats,
            'daily_data': daily_cv_df,
            'problematic_days': problematic_days,
            'weather_analysis': weather_analysis,
            'module_type': self.module_type
        }
        
        print(f"[INFO] CV analysis completed for {self.module_type}:")
        print(f"  - Mean CV error: {summary_stats['mean_cv_error_relative']:.3f}")
        print(f"  - CV correlation: {summary_stats['cv_correlation']:.3f}")
        print(f"  - Problematic days: {len(problematic_days)}/{len(daily_cv_df)}")
        
        return results
    
    def _calculate_cv(self, values: np.ndarray) -> float:
        """Calculate coefficient of variation"""
        if len(values) == 0 or np.mean(values) == 0:
            return 0.0
        return np.std(values) / np.mean(values)
    
    def _expand_timestamps_for_multistep(self, timestamps: pd.DatetimeIndex, 
                                       forecast_steps: int) -> pd.DatetimeIndex:
        """Expand timestamps for multi-step predictions"""
        expanded_timestamps = []
        
        for i, base_time in enumerate(timestamps):
            for step in range(forecast_steps):
                # Assuming 1-hour intervals
                step_time = base_time + timedelta(hours=step)
                expanded_timestamps.append(step_time)
        
        return pd.DatetimeIndex(expanded_timestamps)
    
    def _analyze_cv_weather_dependency(self, daily_cv_df: pd.DataFrame, 
                                     full_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze how CV prediction quality depends on weather conditions"""
        
        # This would require weather data - placeholder for now
        weather_analysis = {
            'sunny_days_cv_error': None,
            'cloudy_days_cv_error': None,
            'high_variability_days_cv_error': None,
            'note': 'Weather dependency analysis requires weather data integration'
        }
        
        return weather_analysis
    
    def create_cv_visualizations(self, cv_results: Dict[str, Any], 
                               save_path: str = None) -> Dict[str, plt.Figure]:
        """
        Create visualizations for CV prediction quality analysis
        
        Args:
            cv_results: Results from analyze_cv_prediction_quality
            save_path: Optional path to save figures
            
        Returns:
            Dictionary of matplotlib figures
        """
        figures = {}
        
        if not cv_results or 'daily_data' not in cv_results:
            print("[WARN] No CV data available for visualization")
            return figures
        
        daily_df = cv_results['daily_data']
        
        # 1. Daily CV Comparison Plot
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        
        ax1.plot(daily_df['date'], daily_df['actual_cv'], 'o-', label='Actual CV', alpha=0.7)
        ax1.plot(daily_df['date'], daily_df['predicted_cv'], 's-', label='Predicted CV', alpha=0.7)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Coefficient of Variation')
        ax1.set_title(f'Daily CV Prediction Quality - {self.module_type.title()} Modules')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        figures['daily_cv_comparison'] = fig1
        
        # 2. CV Error Distribution
        fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Absolute error distribution
        ax2a.hist(daily_df['cv_error_absolute'], bins=20, alpha=0.7, edgecolor='black')
        ax2a.set_xlabel('Absolute CV Error')
        ax2a.set_ylabel('Frequency')
        ax2a.set_title(f'CV Error Distribution - {self.module_type.title()}')
        ax2a.grid(True, alpha=0.3)
        
        # Relative error distribution
        ax2b.hist(daily_df['cv_error_relative'], bins=20, alpha=0.7, edgecolor='black')
        ax2b.set_xlabel('Relative CV Error')
        ax2b.set_ylabel('Frequency')
        ax2b.set_title(f'Relative CV Error Distribution - {self.module_type.title()}')
        ax2b.grid(True, alpha=0.3)
        
        figures['cv_error_distribution'] = fig2
        
        # 3. CV Correlation Scatter Plot
        fig3, ax3 = plt.subplots(figsize=(8, 8))
        
        ax3.scatter(daily_df['actual_cv'], daily_df['predicted_cv'], alpha=0.7)
        
        # Add perfect prediction line
        min_cv = min(daily_df['actual_cv'].min(), daily_df['predicted_cv'].min())
        max_cv = max(daily_df['actual_cv'].max(), daily_df['predicted_cv'].max())
        ax3.plot([min_cv, max_cv], [min_cv, max_cv], 'r--', label='Perfect Prediction')
        
        ax3.set_xlabel('Actual CV')
        ax3.set_ylabel('Predicted CV')
        ax3.set_title(f'CV Prediction Correlation - {self.module_type.title()}')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        corr = daily_df['actual_cv'].corr(daily_df['predicted_cv'])
        ax3.text(0.05, 0.95, f'R = {corr:.3f}', transform=ax3.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        figures['cv_correlation'] = fig3
        
        # Save figures if path provided
        if save_path:
            for name, fig in figures.items():
                fig.savefig(f"{save_path}/cv_analysis_{name}_{self.module_type}.png", 
                           dpi=300, bbox_inches='tight')
        
        return figures


def analyze_cv_prediction_quality(predictions: np.ndarray, 
                                actuals: np.ndarray, 
                                timestamps: pd.DatetimeIndex,
                                module_type: str = 'silicon') -> Dict[str, Any]:
    """
    Convenience function to analyze CV prediction quality
    
    Args:
        predictions: Model predictions
        actuals: Actual values  
        timestamps: Corresponding timestamps
        module_type: 'silicon' or 'perovskite'
        
    Returns:
        CV prediction quality analysis results
    """
    analyzer = VariabilityAnalysis(module_type)
    return analyzer.analyze_cv_prediction_quality(predictions, actuals, timestamps)


if __name__ == "__main__":
    # Example usage
    print("Analytical Package 1: Variability Analysis")
    print("This package analyzes CV prediction quality for PV forecasting models")
