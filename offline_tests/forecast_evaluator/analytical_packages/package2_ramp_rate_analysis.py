"""
Analytical Package 2: Ramp-Rate Analysis
========================================

This package analyzes the prediction quality of LSTM models for ramp-rate
characteristics of Silicon vs. Perovskite modules.

Focus: How well can the LSTM model predict rapid power changes (ramp-rates)
of different module types, not just the general module differences.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta


class RampRateAnalysis:
    """
    Analytical Package 2: Ramp-Rate Analysis for PV Forecasting
    
    Analyzes how well LSTM models can predict ramp-rate characteristics
    of Silicon vs. Perovskite modules.
    """
    
    def __init__(self, module_type: str):
        """
        Initialize Ramp-Rate Analysis for specific module type
        
        Args:
            module_type: 'silicon' or 'perovskite'
        """
        self.module_type = module_type
        self.results = {}
        
    def analyze_ramp_rate_prediction_quality(self, predictions: np.ndarray, 
                                           actuals: np.ndarray, 
                                           timestamps: pd.DatetimeIndex) -> Dict[str, Any]:
        """
        Analyze how well the model predicts ramp-rates (rapid power changes)
        
        Args:
            predictions: Model predictions (n_samples, forecast_steps, n_features)
            actuals: Actual values (n_samples, forecast_steps, n_features)
            timestamps: Corresponding timestamps
            
        Returns:
            Dictionary with ramp-rate prediction quality analysis
        """
        print(f"[INFO] Analyzing ramp-rate prediction quality for {self.module_type} modules...")
        
        # Flatten predictions and actuals for analysis
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
            
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame({
            'timestamp': timestamps_flat,
            'predicted': predictions_flat.flatten(),
            'actual': actuals_flat.flatten()
        })
        
        # Sort by timestamp to ensure correct order
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Calculate ramp-rates (absolute power changes between consecutive time steps)
        df['predicted_ramp_rate'] = df['predicted'].diff().abs()
        df['actual_ramp_rate'] = df['actual'].diff().abs()
        
        # Remove first row (NaN due to diff)
        df = df.dropna().reset_index(drop=True)
        
        if len(df) < 2:
            print(f"[WARN] Insufficient data for ramp-rate analysis")
            return {}
        
        # Calculate daily ramp-rate statistics
        df['date'] = df['timestamp'].dt.date
        daily_ramp_data = []
        
        for date in df['date'].unique():
            day_data = df[df['date'] == date]
            
            if len(day_data) < 2:  # Need at least 2 points for ramp-rate calculation
                continue
                
            # Calculate ramp-rate statistics for this day
            pred_ramp_stats = self._calculate_ramp_rate_stats(day_data['predicted_ramp_rate'].values)
            actual_ramp_stats = self._calculate_ramp_rate_stats(day_data['actual_ramp_rate'].values)
            
            # Calculate ramp-rate prediction errors
            ramp_error_mean = abs(pred_ramp_stats['mean'] - actual_ramp_stats['mean'])
            ramp_error_std = abs(pred_ramp_stats['std'] - actual_ramp_stats['std'])
            ramp_error_max = abs(pred_ramp_stats['max'] - actual_ramp_stats['max'])
            
            # Relative errors
            ramp_error_mean_rel = ramp_error_mean / actual_ramp_stats['mean'] if actual_ramp_stats['mean'] > 0 else 0
            ramp_error_std_rel = ramp_error_std / actual_ramp_stats['std'] if actual_ramp_stats['std'] > 0 else 0
            ramp_error_max_rel = ramp_error_max / actual_ramp_stats['max'] if actual_ramp_stats['max'] > 0 else 0
            
            daily_ramp_data.append({
                'date': date,
                'predicted_mean_ramp': pred_ramp_stats['mean'],
                'actual_mean_ramp': actual_ramp_stats['mean'],
                'predicted_std_ramp': pred_ramp_stats['std'],
                'actual_std_ramp': actual_ramp_stats['std'],
                'predicted_max_ramp': pred_ramp_stats['max'],
                'actual_max_ramp': actual_ramp_stats['max'],
                'ramp_error_mean_abs': ramp_error_mean,
                'ramp_error_std_abs': ramp_error_std,
                'ramp_error_max_abs': ramp_error_max,
                'ramp_error_mean_rel': ramp_error_mean_rel,
                'ramp_error_std_rel': ramp_error_std_rel,
                'ramp_error_max_rel': ramp_error_max_rel,
                'n_points': len(day_data)
            })
        
        daily_ramp_df = pd.DataFrame(daily_ramp_data)
        
        if len(daily_ramp_df) == 0:
            print(f"[WARN] No valid daily data for ramp-rate analysis")
            return {}
        
        # Calculate summary statistics
        summary_stats = {
            'mean_ramp_error_mean_abs': daily_ramp_df['ramp_error_mean_abs'].mean(),
            'std_ramp_error_mean_abs': daily_ramp_df['ramp_error_mean_abs'].std(),
            'mean_ramp_error_mean_rel': daily_ramp_df['ramp_error_mean_rel'].mean(),
            'std_ramp_error_mean_rel': daily_ramp_df['ramp_error_mean_rel'].std(),
            'mean_ramp_error_std_abs': daily_ramp_df['ramp_error_std_abs'].mean(),
            'mean_ramp_error_std_rel': daily_ramp_df['ramp_error_std_rel'].mean(),
            'mean_ramp_error_max_abs': daily_ramp_df['ramp_error_max_abs'].mean(),
            'mean_ramp_error_max_rel': daily_ramp_df['ramp_error_max_rel'].mean(),
            'mean_predicted_ramp': daily_ramp_df['predicted_mean_ramp'].mean(),
            'mean_actual_ramp': daily_ramp_df['actual_mean_ramp'].mean(),
            'ramp_correlation_mean': daily_ramp_df['predicted_mean_ramp'].corr(daily_ramp_df['actual_mean_ramp']),
            'ramp_correlation_std': daily_ramp_df['predicted_std_ramp'].corr(daily_ramp_df['actual_std_ramp']),
            'ramp_correlation_max': daily_ramp_df['predicted_max_ramp'].corr(daily_ramp_df['actual_max_ramp']),
            'n_days': len(daily_ramp_df)
        }
        
        # Identify problematic days (high ramp-rate prediction errors)
        threshold_high_error = daily_ramp_df['ramp_error_mean_rel'].quantile(0.75)
        problematic_days = daily_ramp_df[daily_ramp_df['ramp_error_mean_rel'] > threshold_high_error]
        
        # Analyze ramp-rate patterns
        ramp_patterns = self._analyze_ramp_patterns(daily_ramp_df, df)
        
        # Weather condition analysis (if available)
        weather_analysis = self._analyze_ramp_weather_dependency(daily_ramp_df, df)
        
        results = {
            'summary_stats': summary_stats,
            'daily_data': daily_ramp_df,
            'problematic_days': problematic_days,
            'ramp_patterns': ramp_patterns,
            'weather_analysis': weather_analysis,
            'module_type': self.module_type
        }
        
        print(f"[INFO] Ramp-rate analysis completed for {self.module_type}:")
        print(f"  - Mean ramp error (relative): {summary_stats['mean_ramp_error_mean_rel']:.3f}")
        print(f"  - Ramp correlation (mean): {summary_stats['ramp_correlation_mean']:.3f}")
        print(f"  - Ramp correlation (std): {summary_stats['ramp_correlation_std']:.3f}")
        print(f"  - Problematic days: {len(problematic_days)}/{len(daily_ramp_df)}")
        
        return results
    
    def _calculate_ramp_rate_stats(self, ramp_rates: np.ndarray) -> Dict[str, float]:
        """Calculate statistics for ramp-rates"""
        if len(ramp_rates) == 0:
            return {'mean': 0, 'std': 0, 'max': 0, 'min': 0, 'median': 0}
        
        return {
            'mean': np.mean(ramp_rates),
            'std': np.std(ramp_rates),
            'max': np.max(ramp_rates),
            'min': np.min(ramp_rates),
            'median': np.median(ramp_rates)
        }
    
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
    
    def _analyze_ramp_patterns(self, daily_ramp_df: pd.DataFrame, 
                             full_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze ramp-rate patterns and characteristics"""
        
        # Analyze ramp-rate distribution
        all_actual_ramps = full_df['actual_ramp_rate'].values
        all_predicted_ramps = full_df['predicted_ramp_rate'].values
        
        # Categorize ramp-rates
        ramp_categories = {
            'low_ramp': (all_actual_ramps < np.percentile(all_actual_ramps, 33)),
            'medium_ramp': ((all_actual_ramps >= np.percentile(all_actual_ramps, 33)) & 
                           (all_actual_ramps < np.percentile(all_actual_ramps, 67))),
            'high_ramp': (all_actual_ramps >= np.percentile(all_actual_ramps, 67))
        }
        
        # Calculate prediction errors for each category
        category_errors = {}
        for category, mask in ramp_categories.items():
            if np.any(mask):
                actual_ramps = all_actual_ramps[mask]
                predicted_ramps = all_predicted_ramps[mask]
                errors = np.abs(predicted_ramps - actual_ramps)
                
                category_errors[category] = {
                    'mean_error': np.mean(errors),
                    'std_error': np.std(errors),
                    'n_points': len(errors),
                    'mean_actual': np.mean(actual_ramps),
                    'mean_predicted': np.mean(predicted_ramps)
                }
        
        return {
            'category_errors': category_errors,
            'ramp_percentiles': {
                'p33': np.percentile(all_actual_ramps, 33),
                'p67': np.percentile(all_actual_ramps, 67),
                'p95': np.percentile(all_actual_ramps, 95)
            }
        }
    
    def _analyze_ramp_weather_dependency(self, daily_ramp_df: pd.DataFrame, 
                                       full_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze how ramp-rate prediction quality depends on weather conditions"""
        
        # This would require weather data - placeholder for now
        weather_analysis = {
            'sunny_days_ramp_error': None,
            'cloudy_days_ramp_error': None,
            'high_variability_days_ramp_error': None,
            'note': 'Weather dependency analysis requires weather data integration'
        }
        
        return weather_analysis
    
    def create_ramp_rate_visualizations(self, ramp_results: Dict[str, Any], 
                                      save_path: str = None) -> Dict[str, plt.Figure]:
        """
        Create visualizations for ramp-rate prediction quality analysis
        
        Args:
            ramp_results: Results from analyze_ramp_rate_prediction_quality
            save_path: Optional path to save figures
            
        Returns:
            Dictionary of matplotlib figures
        """
        figures = {}
        
        if not ramp_results or 'daily_data' not in ramp_results:
            print("[WARN] No ramp-rate data available for visualization")
            return figures
        
        daily_df = ramp_results['daily_data']
        
        # 1. Daily Ramp-Rate Comparison Plot
        fig1, (ax1a, ax1b, ax1c) = plt.subplots(3, 1, figsize=(12, 15))
        
        # Mean ramp-rates
        ax1a.plot(daily_df['date'], daily_df['actual_mean_ramp'], 'o-', label='Actual Mean Ramp', alpha=0.7)
        ax1a.plot(daily_df['date'], daily_df['predicted_mean_ramp'], 's-', label='Predicted Mean Ramp', alpha=0.7)
        ax1a.set_xlabel('Date')
        ax1a.set_ylabel('Mean Ramp-Rate')
        ax1a.set_title(f'Daily Mean Ramp-Rate Prediction - {self.module_type.title()} Modules')
        ax1a.legend()
        ax1a.grid(True, alpha=0.3)
        
        # Std ramp-rates
        ax1b.plot(daily_df['date'], daily_df['actual_std_ramp'], 'o-', label='Actual Std Ramp', alpha=0.7)
        ax1b.plot(daily_df['date'], daily_df['predicted_std_ramp'], 's-', label='Predicted Std Ramp', alpha=0.7)
        ax1b.set_xlabel('Date')
        ax1b.set_ylabel('Std Ramp-Rate')
        ax1b.set_title(f'Daily Std Ramp-Rate Prediction - {self.module_type.title()} Modules')
        ax1b.legend()
        ax1b.grid(True, alpha=0.3)
        
        # Max ramp-rates
        ax1c.plot(daily_df['date'], daily_df['actual_max_ramp'], 'o-', label='Actual Max Ramp', alpha=0.7)
        ax1c.plot(daily_df['date'], daily_df['predicted_max_ramp'], 's-', label='Predicted Max Ramp', alpha=0.7)
        ax1c.set_xlabel('Date')
        ax1c.set_ylabel('Max Ramp-Rate')
        ax1c.set_title(f'Daily Max Ramp-Rate Prediction - {self.module_type.title()} Modules')
        ax1c.legend()
        ax1c.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        figures['daily_ramp_comparison'] = fig1
        
        # 2. Ramp-Rate Error Distribution
        fig2, (ax2a, ax2b, ax2c) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Mean error distribution
        ax2a.hist(daily_df['ramp_error_mean_rel'], bins=20, alpha=0.7, edgecolor='black')
        ax2a.set_xlabel('Relative Mean Ramp Error')
        ax2a.set_ylabel('Frequency')
        ax2a.set_title(f'Mean Ramp Error Distribution - {self.module_type.title()}')
        ax2a.grid(True, alpha=0.3)
        
        # Std error distribution
        ax2b.hist(daily_df['ramp_error_std_rel'], bins=20, alpha=0.7, edgecolor='black')
        ax2b.set_xlabel('Relative Std Ramp Error')
        ax2b.set_ylabel('Frequency')
        ax2b.set_title(f'Std Ramp Error Distribution - {self.module_type.title()}')
        ax2b.grid(True, alpha=0.3)
        
        # Max error distribution
        ax2c.hist(daily_df['ramp_error_max_rel'], bins=20, alpha=0.7, edgecolor='black')
        ax2c.set_xlabel('Relative Max Ramp Error')
        ax2c.set_ylabel('Frequency')
        ax2c.set_title(f'Max Ramp Error Distribution - {self.module_type.title()}')
        ax2c.grid(True, alpha=0.3)
        
        figures['ramp_error_distribution'] = fig2
        
        # 3. Ramp-Rate Correlation Scatter Plots
        fig3, (ax3a, ax3b, ax3c) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Mean ramp correlation
        ax3a.scatter(daily_df['actual_mean_ramp'], daily_df['predicted_mean_ramp'], alpha=0.7)
        min_val = min(daily_df['actual_mean_ramp'].min(), daily_df['predicted_mean_ramp'].min())
        max_val = max(daily_df['actual_mean_ramp'].max(), daily_df['predicted_mean_ramp'].max())
        ax3a.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        ax3a.set_xlabel('Actual Mean Ramp')
        ax3a.set_ylabel('Predicted Mean Ramp')
        ax3a.set_title(f'Mean Ramp Correlation - {self.module_type.title()}')
        ax3a.legend()
        ax3a.grid(True, alpha=0.3)
        corr_mean = daily_df['actual_mean_ramp'].corr(daily_df['predicted_mean_ramp'])
        ax3a.text(0.05, 0.95, f'R = {corr_mean:.3f}', transform=ax3a.transAxes, 
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Std ramp correlation
        ax3b.scatter(daily_df['actual_std_ramp'], daily_df['predicted_std_ramp'], alpha=0.7)
        min_val = min(daily_df['actual_std_ramp'].min(), daily_df['predicted_std_ramp'].min())
        max_val = max(daily_df['actual_std_ramp'].max(), daily_df['predicted_std_ramp'].max())
        ax3b.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        ax3b.set_xlabel('Actual Std Ramp')
        ax3b.set_ylabel('Predicted Std Ramp')
        ax3b.set_title(f'Std Ramp Correlation - {self.module_type.title()}')
        ax3b.legend()
        ax3b.grid(True, alpha=0.3)
        corr_std = daily_df['actual_std_ramp'].corr(daily_df['predicted_std_ramp'])
        ax3b.text(0.05, 0.95, f'R = {corr_std:.3f}', transform=ax3b.transAxes, 
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Max ramp correlation
        ax3c.scatter(daily_df['actual_max_ramp'], daily_df['predicted_max_ramp'], alpha=0.7)
        min_val = min(daily_df['actual_max_ramp'].min(), daily_df['predicted_max_ramp'].min())
        max_val = max(daily_df['actual_max_ramp'].max(), daily_df['predicted_max_ramp'].max())
        ax3c.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        ax3c.set_xlabel('Actual Max Ramp')
        ax3c.set_ylabel('Predicted Max Ramp')
        ax3c.set_title(f'Max Ramp Correlation - {self.module_type.title()}')
        ax3c.legend()
        ax3c.grid(True, alpha=0.3)
        corr_max = daily_df['actual_max_ramp'].corr(daily_df['predicted_max_ramp'])
        ax3c.text(0.05, 0.95, f'R = {corr_max:.3f}', transform=ax3c.transAxes, 
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        figures['ramp_correlations'] = fig3
        
        # Save figures if path provided
        if save_path:
            for name, fig in figures.items():
                fig.savefig(f"{save_path}/ramp_analysis_{name}_{self.module_type}.png", 
                           dpi=300, bbox_inches='tight')
        
        return figures


def analyze_ramp_rate_prediction_quality(predictions: np.ndarray, 
                                       actuals: np.ndarray, 
                                       timestamps: pd.DatetimeIndex,
                                       module_type: str = 'silicon') -> Dict[str, Any]:
    """
    Convenience function to analyze ramp-rate prediction quality
    
    Args:
        predictions: Model predictions
        actuals: Actual values  
        timestamps: Corresponding timestamps
        module_type: 'silicon' or 'perovskite'
        
    Returns:
        Ramp-rate prediction quality analysis results
    """
    analyzer = RampRateAnalysis(module_type)
    return analyzer.analyze_ramp_rate_prediction_quality(predictions, actuals, timestamps)


if __name__ == "__main__":
    # Example usage
    print("Analytical Package 2: Ramp-Rate Analysis")
    print("This package analyzes ramp-rate prediction quality for PV forecasting models")
