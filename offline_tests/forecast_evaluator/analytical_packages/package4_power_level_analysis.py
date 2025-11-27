"""
Analytical Package 4: Power-Level Analysis
==========================================

This package analyzes prediction quality across different power levels
to identify where Silicon vs. Perovskite differences are most pronounced.

Focus: How well can the LSTM model predict different power ranges,
and where do Si vs. Pvk show the most significant differences.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy import stats


class PowerLevelAnalysis:
    """
    Analytical Package 4: Power-Level Analysis for PV Forecasting
    
    Analyzes prediction quality across different power levels to identify
    where Silicon vs. Perovskite differences are most pronounced.
    """
    
    def __init__(self, module_type: str):
        """
        Initialize Power-Level Analysis for specific module type
        
        Args:
            module_type: 'silicon' or 'perovskite'
        """
        self.module_type = module_type
        self.results = {}
        
    def analyze_power_level_prediction_quality(self, predictions: np.ndarray, 
                                             actuals: np.ndarray, 
                                             timestamps: pd.DatetimeIndex) -> Dict[str, Any]:
        """
        Analyze prediction quality across different power levels
        
        Args:
            predictions: Model predictions (n_samples, forecast_steps, n_features)
            actuals: Actual values (n_samples, forecast_steps, n_features)
            timestamps: Corresponding timestamps
            
        Returns:
            Dictionary with power-level prediction quality analysis
        """
        print(f"[INFO] Analyzing power-level prediction quality for {self.module_type} modules...")
        
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
        
        # Define power level categories
        power_levels = self._define_power_levels(df['actual'])
        
        # Analyze each power level
        level_analysis = {}
        for level_name, level_mask in power_levels.items():
            if level_mask.sum() > 10:  # Need sufficient data points
                level_data = df[level_mask]
                level_metrics = self._analyze_power_level(level_data, level_name)
                level_analysis[level_name] = level_metrics
        
        # Compare power levels
        level_comparison = self._compare_power_levels(level_analysis)
        
        # Analyze temporal patterns within power levels
        temporal_analysis = self._analyze_temporal_patterns(df, power_levels)
        
        # Analyze prediction errors by power level
        error_analysis = self._analyze_prediction_errors(df, power_levels)
        
        results = {
            'power_levels': power_levels,
            'level_analysis': level_analysis,
            'level_comparison': level_comparison,
            'temporal_analysis': temporal_analysis,
            'error_analysis': error_analysis,
            'module_type': self.module_type
        }
        
        print(f"[INFO] Power-level analysis completed for {self.module_type}:")
        print(f"  - Power levels analyzed: {len(level_analysis)}")
        print(f"  - Total data points: {len(df)}")
        print(f"  - Data distribution across levels:")
        for level_name, level_mask in power_levels.items():
            count = level_mask.sum()
            percentage = count / len(df) * 100
            print(f"    {level_name}: {count} points ({percentage:.1f}%)")
        
        return results
    
    def _define_power_levels(self, actual_values: pd.Series) -> Dict[str, pd.Series]:
        """Define power level categories based on quantiles"""
        
        # Define power level thresholds
        thresholds = {
            'very_low': (0.0, 0.1),      # 0-10% of max power
            'low': (0.1, 0.3),           # 10-30% of max power
            'medium': (0.3, 0.6),        # 30-60% of max power
            'high': (0.6, 0.8),          # 60-80% of max power
            'very_high': (0.8, 1.0),     # 80-100% of max power
            'peak': (0.9, 1.0)           # 90-100% of max power (overlap with very_high)
        }
        
        power_levels = {}
        for level_name, (min_thresh, max_thresh) in thresholds.items():
            mask = (actual_values >= min_thresh) & (actual_values < max_thresh)
            power_levels[level_name] = mask
            
        return power_levels
    
    def _analyze_power_level(self, level_data: pd.DataFrame, level_name: str) -> Dict[str, Any]:
        """Analyze prediction quality for a specific power level"""
        
        if len(level_data) < 5:
            return {'error': 'Insufficient data'}
        
        # Calculate basic metrics
        actual = level_data['actual'].values
        predicted = level_data['predicted'].values
        
        # Calculate prediction errors
        errors = predicted - actual
        abs_errors = np.abs(errors)
        rel_errors = abs_errors / (actual + 1e-8)  # Avoid division by zero
        
        # Calculate metrics
        metrics = {
            'n_points': len(level_data),
            'actual_mean': np.mean(actual),
            'actual_std': np.std(actual),
            'actual_min': np.min(actual),
            'actual_max': np.max(actual),
            'predicted_mean': np.mean(predicted),
            'predicted_std': np.std(predicted),
            'predicted_min': np.min(predicted),
            'predicted_max': np.max(predicted),
            'mae': np.mean(abs_errors),
            'rmse': np.sqrt(np.mean(errors**2)),
            'mape': np.mean(rel_errors) * 100,
            'r2': self._calculate_r2(actual, predicted),
            'correlation': np.corrcoef(actual, predicted)[0, 1],
            'bias': np.mean(errors),
            'std_error': np.std(errors)
        }
        
        # Calculate additional power-level specific metrics
        metrics.update({
            'peak_detection_accuracy': self._calculate_peak_detection_accuracy(actual, predicted),
            'ramp_detection_accuracy': self._calculate_ramp_detection_accuracy(actual, predicted),
            'stability_score': self._calculate_stability_score(actual, predicted)
        })
        
        return metrics
    
    def _calculate_r2(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate R² score"""
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    def _calculate_peak_detection_accuracy(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate how well the model detects peaks"""
        # Define peaks as values above 80th percentile
        peak_threshold = np.percentile(actual, 80)
        actual_peaks = actual > peak_threshold
        predicted_peaks = predicted > peak_threshold
        
        # Calculate accuracy
        correct_peaks = np.sum(actual_peaks & predicted_peaks)
        total_peaks = np.sum(actual_peaks)
        
        return correct_peaks / total_peaks if total_peaks > 0 else 0
    
    def _calculate_ramp_detection_accuracy(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate how well the model detects ramps (rapid changes)"""
        # Calculate ramps as large changes between consecutive points
        actual_ramps = np.abs(np.diff(actual))
        predicted_ramps = np.abs(np.diff(predicted))
        
        # Define significant ramps as above 75th percentile
        ramp_threshold = np.percentile(actual_ramps, 75)
        actual_significant_ramps = actual_ramps > ramp_threshold
        predicted_significant_ramps = predicted_ramps > ramp_threshold
        
        # Calculate accuracy
        correct_ramps = np.sum(actual_significant_ramps & predicted_significant_ramps)
        total_ramps = np.sum(actual_significant_ramps)
        
        return correct_ramps / total_ramps if total_ramps > 0 else 0
    
    def _calculate_stability_score(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate how stable the predictions are relative to actuals"""
        # Calculate coefficient of variation for both
        actual_cv = np.std(actual) / (np.mean(actual) + 1e-8)
        predicted_cv = np.std(predicted) / (np.mean(predicted) + 1e-8)
        
        # Stability score is inverse of the difference in CV
        cv_diff = abs(actual_cv - predicted_cv)
        return 1 / (1 + cv_diff) if cv_diff > 0 else 1
    
    def _compare_power_levels(self, level_analysis: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compare prediction quality across power levels"""
        
        comparison = {
            'best_performing_level': None,
            'worst_performing_level': None,
            'level_rankings': {},
            'performance_gaps': {}
        }
        
        # Rank levels by R² score
        r2_scores = {}
        for level_name, metrics in level_analysis.items():
            if 'error' not in metrics:
                r2_scores[level_name] = metrics.get('r2', 0)
        
        if r2_scores:
            sorted_levels = sorted(r2_scores.items(), key=lambda x: x[1], reverse=True)
            comparison['level_rankings'] = dict(sorted_levels)
            comparison['best_performing_level'] = sorted_levels[0][0]
            comparison['worst_performing_level'] = sorted_levels[-1][0]
            
            # Calculate performance gaps
            best_r2 = sorted_levels[0][1]
            for level_name, r2 in r2_scores.items():
                comparison['performance_gaps'][level_name] = best_r2 - r2
        
        return comparison
    
    def _analyze_temporal_patterns(self, df: pd.DataFrame, power_levels: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Analyze temporal patterns within power levels"""
        
        temporal_analysis = {}
        
        for level_name, level_mask in power_levels.items():
            if level_mask.sum() < 10:
                continue
                
            level_data = df[level_mask].copy()
            level_data['hour'] = level_data['timestamp'].dt.hour
            
            # Analyze by hour of day
            hourly_metrics = {}
            for hour in range(24):
                hour_data = level_data[level_data['hour'] == hour]
                if len(hour_data) > 3:
                    hourly_metrics[hour] = {
                        'n_points': len(hour_data),
                        'mae': np.mean(np.abs(hour_data['predicted'] - hour_data['actual'])),
                        'r2': self._calculate_r2(hour_data['actual'].values, hour_data['predicted'].values)
                    }
            
            temporal_analysis[level_name] = {
                'hourly_metrics': hourly_metrics,
                'best_hour': max(hourly_metrics.keys(), key=lambda h: hourly_metrics[h]['r2']) if hourly_metrics else None,
                'worst_hour': min(hourly_metrics.keys(), key=lambda h: hourly_metrics[h]['r2']) if hourly_metrics else None
            }
        
        return temporal_analysis
    
    def _analyze_prediction_errors(self, df: pd.DataFrame, power_levels: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Analyze prediction errors by power level"""
        
        error_analysis = {}
        
        for level_name, level_mask in power_levels.items():
            if level_mask.sum() < 10:
                continue
                
            level_data = df[level_mask]
            errors = level_data['predicted'] - level_data['actual']
            
            error_analysis[level_name] = {
                'mean_error': np.mean(errors),
                'std_error': np.std(errors),
                'error_skewness': stats.skew(errors),
                'error_kurtosis': stats.kurtosis(errors),
                'overprediction_rate': np.sum(errors > 0) / len(errors),
                'underprediction_rate': np.sum(errors < 0) / len(errors),
                'large_error_rate': np.sum(np.abs(errors) > 0.2) / len(errors)
            }
        
        return error_analysis
    
    def _expand_timestamps_for_multistep(self, timestamps: pd.DatetimeIndex, 
                                       forecast_steps: int) -> pd.DatetimeIndex:
        """Expand timestamps for multi-step predictions"""
        expanded_timestamps = []
        
        for i, base_time in enumerate(timestamps):
            for step in range(forecast_steps):
                # Assuming 10-minute intervals
                step_time = base_time + timedelta(minutes=step * 10)
                expanded_timestamps.append(step_time)
        
        return pd.DatetimeIndex(expanded_timestamps)
    
    def create_power_level_visualizations(self, power_results: Dict[str, Any], 
                                        save_path: str = None) -> Dict[str, plt.Figure]:
        """
        Create visualizations for power-level analysis
        
        Args:
            power_results: Results from analyze_power_level_prediction_quality
            save_path: Optional path to save figures
            
        Returns:
            Dictionary of matplotlib figures
        """
        figures = {}
        
        if not power_results or 'level_analysis' not in power_results:
            print("[WARN] No power-level data available for visualization")
            return figures
        
        level_analysis = power_results['level_analysis']
        
        # 1. Power Level Performance Comparison
        fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(15, 6))
        
        # R² scores by power level
        levels = list(level_analysis.keys())
        r2_scores = [level_analysis[level].get('r2', 0) for level in levels]
        mae_scores = [level_analysis[level].get('mae', 0) for level in levels]
        
        ax1a.bar(levels, r2_scores, alpha=0.7, color='skyblue')
        ax1a.set_xlabel('Power Level')
        ax1a.set_ylabel('R² Score')
        ax1a.set_title(f'R² Score by Power Level - {self.module_type.title()} Modules')
        ax1a.tick_params(axis='x', rotation=45)
        ax1a.grid(True, alpha=0.3)
        
        ax1b.bar(levels, mae_scores, alpha=0.7, color='lightcoral')
        ax1b.set_xlabel('Power Level')
        ax1b.set_ylabel('MAE')
        ax1b.set_title(f'MAE by Power Level - {self.module_type.title()} Modules')
        ax1b.tick_params(axis='x', rotation=45)
        ax1b.grid(True, alpha=0.3)
        
        figures['power_level_performance'] = fig1
        
        # 2. Prediction vs Actual by Power Level
        fig2, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, (level_name, metrics) in enumerate(level_analysis.items()):
            if i >= 6:  # Limit to 6 subplots
                break
                
            if 'error' in metrics:
                continue
                
            # Get data for this power level
            level_mask = power_results['power_levels'][level_name]
            level_data = power_results.get('level_data', pd.DataFrame())
            
            if len(level_data) > 0:
                level_subset = level_data[level_mask]
                
                ax = axes[i]
                ax.scatter(level_subset['actual'], level_subset['predicted'], alpha=0.6, s=20)
                
                # Add perfect prediction line
                min_val = min(level_subset['actual'].min(), level_subset['predicted'].min())
                max_val = max(level_subset['actual'].max(), level_subset['predicted'].max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
                
                ax.set_xlabel('Actual')
                ax.set_ylabel('Predicted')
                ax.set_title(f'{level_name.title()} Level\nR² = {metrics.get("r2", 0):.3f}')
                ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(level_analysis), 6):
            axes[i].set_visible(False)
        
        figures['power_level_scatter'] = fig2
        
        # 3. Error Distribution by Power Level
        fig3, ax = plt.subplots(figsize=(12, 8))
        
        error_data = []
        level_labels = []
        
        for level_name, metrics in level_analysis.items():
            if 'error' not in metrics and level_name in power_results['power_levels']:
                level_mask = power_results['power_levels'][level_name]
                if 'level_data' in power_results:
                    level_subset = power_results['level_data'][level_mask]
                    errors = level_subset['predicted'] - level_subset['actual']
                    error_data.append(errors)
                    level_labels.append(level_name)
        
        if error_data:
            ax.boxplot(error_data, labels=level_labels)
            ax.set_xlabel('Power Level')
            ax.set_ylabel('Prediction Error')
            ax.set_title(f'Error Distribution by Power Level - {self.module_type.title()} Modules')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.7)
        
        figures['error_distribution'] = fig3
        
        # Save figures if path provided
        if save_path:
            for name, fig in figures.items():
                fig.savefig(f"{save_path}/power_level_analysis_{name}_{self.module_type}.png", 
                           dpi=300, bbox_inches='tight')
        
        return figures


def analyze_power_level_prediction_quality(predictions: np.ndarray, 
                                         actuals: np.ndarray, 
                                         timestamps: pd.DatetimeIndex,
                                         module_type: str = 'silicon') -> Dict[str, Any]:
    """
    Convenience function to analyze power-level prediction quality
    
    Args:
        predictions: Model predictions
        actuals: Actual values  
        timestamps: Corresponding timestamps
        module_type: 'silicon' or 'perovskite'
        
    Returns:
        Power-level prediction quality analysis results
    """
    analyzer = PowerLevelAnalysis(module_type)
    return analyzer.analyze_power_level_prediction_quality(predictions, actuals, timestamps)


if __name__ == "__main__":
    # Example usage
    print("Analytical Package 4: Power-Level Analysis")
    print("This package analyzes prediction quality across different power levels")
