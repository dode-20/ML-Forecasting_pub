"""
Analytical Package 3: Intermittency Analysis
============================================

This package analyzes the prediction quality of LSTM models for intermittency
characteristics of Silicon vs. Perovskite modules.

Focus: How well can the LSTM model predict intermittent power behavior
(rapid on/off transitions, cloud-induced fluctuations) of different module types.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy import stats
from scipy.signal import find_peaks


class IntermittencyAnalysis:
    """
    Analytical Package 3: Intermittency Analysis for PV Forecasting
    
    Analyzes how well LSTM models can predict intermittent power behavior
    of Silicon vs. Perovskite modules.
    """
    
    def __init__(self, module_type: str):
        """
        Initialize Intermittency Analysis for specific module type
        
        Args:
            module_type: 'silicon' or 'perovskite'
        """
        self.module_type = module_type
        self.results = {}
        
    def analyze_intermittency_prediction_quality(self, predictions: np.ndarray, 
                                               actuals: np.ndarray, 
                                               timestamps: pd.DatetimeIndex) -> Dict[str, Any]:
        """
        Analyze how well the model predicts intermittency characteristics
        
        Args:
            predictions: Model predictions (n_samples, forecast_steps, n_features)
            actuals: Actual values (n_samples, forecast_steps, n_features)
            timestamps: Corresponding timestamps
            
        Returns:
            Dictionary with intermittency prediction quality analysis
        """
        print(f"[INFO] Analyzing intermittency prediction quality for {self.module_type} modules...")
        
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
        
        # Calculate intermittency metrics
        intermittency_metrics = self._calculate_intermittency_metrics(df)
        
        # Analyze daily intermittency patterns
        daily_intermittency = self._analyze_daily_intermittency(df)
        
        # Analyze transition patterns (on/off states)
        transition_analysis = self._analyze_transition_patterns(df)
        
        # Analyze cloud-induced fluctuations
        cloud_fluctuation_analysis = self._analyze_cloud_fluctuations(df)
        
        # Calculate prediction quality metrics for intermittency
        prediction_quality = self._calculate_intermittency_prediction_quality(
            daily_intermittency, intermittency_metrics
        )
        
        # Add transition and cloud fluctuation metrics to prediction_quality
        prediction_quality['transition_accuracy'] = transition_analysis.get('transition_accuracy', 0)
        prediction_quality['cloud_fluctuation_correlation'] = cloud_fluctuation_analysis.get('fluctuation_correlation', 0)
        
        # Identify problematic periods
        problematic_periods = self._identify_problematic_periods(df, daily_intermittency)
        
        results = {
            'intermittency_metrics': intermittency_metrics,
            'daily_intermittency': daily_intermittency,
            'transition_analysis': transition_analysis,
            'cloud_fluctuation_analysis': cloud_fluctuation_analysis,
            'prediction_quality': prediction_quality,
            'problematic_periods': problematic_periods,
            'module_type': self.module_type
        }
        
        print(f"[INFO] Intermittency analysis completed for {self.module_type}:")
        print(f"  - Mean intermittency error: {prediction_quality.get('mean_intermittency_error', 0):.3f}")
        print(f"  - Transition accuracy: {prediction_quality.get('transition_accuracy', 0):.3f}")
        print(f"  - Cloud fluctuation correlation: {prediction_quality.get('cloud_fluctuation_correlation', 0):.3f}")
        print(f"  - Problematic periods: {len(problematic_periods)}")
        
        return results
    
    def _calculate_intermittency_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate basic intermittency metrics"""
        
        # Power threshold for "on" state (e.g., 10% of max power)
        power_threshold = df['actual'].max() * 0.1
        
        # Identify on/off states
        df['actual_on'] = df['actual'] > power_threshold
        df['predicted_on'] = df['predicted'] > power_threshold
        
        # Calculate intermittency ratio (fraction of time in "on" state)
        actual_intermittency_ratio = df['actual_on'].mean()
        predicted_intermittency_ratio = df['predicted_on'].mean()
        
        # Calculate state persistence (average duration of on/off periods)
        actual_persistence = self._calculate_state_persistence(df['actual_on'])
        predicted_persistence = self._calculate_state_persistence(df['predicted_on'])
        
        # Calculate power variability during "on" periods
        actual_on_power = df[df['actual_on']]['actual']
        predicted_on_power = df[df['predicted_on']]['predicted']
        
        actual_on_variability = actual_on_power.std() if len(actual_on_power) > 1 else 0
        predicted_on_variability = predicted_on_power.std() if len(predicted_on_power) > 1 else 0
        
        return {
            'power_threshold': power_threshold,
            'actual_intermittency_ratio': actual_intermittency_ratio,
            'predicted_intermittency_ratio': predicted_intermittency_ratio,
            'intermittency_ratio_error': abs(actual_intermittency_ratio - predicted_intermittency_ratio),
            'actual_on_duration': actual_persistence['on_duration'],
            'predicted_on_duration': predicted_persistence['on_duration'],
            'actual_off_duration': actual_persistence['off_duration'],
            'predicted_off_duration': predicted_persistence['off_duration'],
            'actual_on_power_std': actual_on_variability,
            'predicted_on_power_std': predicted_on_variability,
            'actual_persistence': actual_persistence,
            'predicted_persistence': predicted_persistence,
            'persistence_error': {
                'on_duration_error': abs(actual_persistence['on_duration'] - predicted_persistence['on_duration']),
                'off_duration_error': abs(actual_persistence['off_duration'] - predicted_persistence['off_duration']),
                'avg_duration_error': abs(actual_persistence['avg_duration'] - predicted_persistence['avg_duration'])
            },
            'actual_on_variability': actual_on_variability,
            'predicted_on_variability': predicted_on_variability,
            'on_variability_error': abs(actual_on_variability - predicted_on_variability)
        }
    
    def _calculate_state_persistence(self, state_series: pd.Series) -> Dict[str, float]:
        """Calculate average duration of on/off states"""
        
        # Find state transitions
        state_changes = state_series.diff().fillna(0) != 0
        transition_indices = state_changes[state_changes].index.tolist()
        
        if len(transition_indices) < 2:
            return {'on_duration': 0, 'off_duration': 0, 'avg_duration': 0}
        
        # Calculate durations between transitions
        durations = []
        current_state = state_series.iloc[0]
        
        for i in range(len(transition_indices) - 1):
            start_idx = transition_indices[i]
            end_idx = transition_indices[i + 1]
            duration = end_idx - start_idx
            durations.append(duration)
            current_state = not current_state
        
        # Add final duration
        final_duration = len(state_series) - transition_indices[-1]
        durations.append(final_duration)
        
        if not durations:
            return {'on_duration': 0, 'off_duration': 0, 'avg_duration': 0}
        
        # Separate on and off durations
        on_durations = []
        off_durations = []
        current_state = state_series.iloc[0]
        
        for i, duration in enumerate(durations):
            if current_state:
                on_durations.append(duration)
            else:
                off_durations.append(duration)
            current_state = not current_state
        
        return {
            'on_duration': np.mean(on_durations) if on_durations else 0,
            'off_duration': np.mean(off_durations) if off_durations else 0,
            'avg_duration': np.mean(durations)
        }
    
    def _analyze_daily_intermittency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze intermittency patterns on a daily basis"""
        
        df['date'] = df['timestamp'].dt.date
        daily_data = []
        
        for date in df['date'].unique():
            day_df = df[df['date'] == date].copy()
            
            if len(day_df) < 10:  # Need sufficient data points
                continue
            
            # Calculate daily intermittency metrics
            power_threshold = day_df['actual'].max() * 0.1
            day_df['actual_on'] = day_df['actual'] > power_threshold
            day_df['predicted_on'] = day_df['predicted'] > power_threshold
            
            # Intermittency ratio
            actual_ratio = day_df['actual_on'].mean()
            predicted_ratio = day_df['predicted_on'].mean()
            
            # State persistence
            actual_persistence = self._calculate_state_persistence(day_df['actual_on'])
            predicted_persistence = self._calculate_state_persistence(day_df['predicted_on'])
            
            # Power variability during on periods
            actual_on_power = day_df[day_df['actual_on']]['actual']
            predicted_on_power = day_df[day_df['predicted_on']]['predicted']
            
            actual_on_std = actual_on_power.std() if len(actual_on_power) > 1 else 0
            predicted_on_std = predicted_on_power.std() if len(predicted_on_power) > 1 else 0
            
            # Number of state transitions
            actual_transitions = (day_df['actual_on'].diff() != 0).sum()
            predicted_transitions = (day_df['predicted_on'].diff() != 0).sum()
            
            daily_data.append({
                'date': date,
                'actual_intermittency_ratio': actual_ratio,
                'predicted_intermittency_ratio': predicted_ratio,
                'intermittency_ratio_error': abs(actual_ratio - predicted_ratio),
                'actual_on_duration': actual_persistence['on_duration'],
                'predicted_on_duration': predicted_persistence['on_duration'],
                'on_duration_error': abs(actual_persistence['on_duration'] - predicted_persistence['on_duration']),
                'actual_off_duration': actual_persistence['off_duration'],
                'predicted_off_duration': predicted_persistence['off_duration'],
                'off_duration_error': abs(actual_persistence['off_duration'] - predicted_persistence['off_duration']),
                'actual_on_std': actual_on_std,
                'predicted_on_std': predicted_on_std,
                'on_std_error': abs(actual_on_std - predicted_on_std),
                'actual_transitions': actual_transitions,
                'predicted_transitions': predicted_transitions,
                'transition_error': abs(actual_transitions - predicted_transitions),
                'n_points': len(day_df)
            })
        
        return pd.DataFrame(daily_data)
    
    def _analyze_transition_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze on/off transition patterns"""
        
        power_threshold = df['actual'].max() * 0.1
        df['actual_on'] = df['actual'] > power_threshold
        df['predicted_on'] = df['predicted'] > power_threshold
        
        # Find actual transitions
        actual_transitions = df['actual_on'].diff() != 0
        predicted_transitions = df['predicted_on'].diff() != 0
        
        # Calculate transition accuracy
        transition_accuracy = (actual_transitions == predicted_transitions).mean()
        
        # Analyze transition timing errors
        actual_transition_times = df[actual_transitions]['timestamp']
        predicted_transition_times = df[predicted_transitions]['timestamp']
        
        # Calculate transition correlation
        transition_correlation = actual_transitions.corr(predicted_transitions.astype(int))
        
        # Analyze transition frequency
        actual_transition_freq = actual_transitions.sum() / len(df)
        predicted_transition_freq = predicted_transitions.sum() / len(df)
        
        return {
            'transition_accuracy': transition_accuracy,
            'transition_correlation': transition_correlation,
            'actual_transition_frequency': actual_transition_freq,
            'predicted_transition_frequency': predicted_transition_freq,
            'transition_frequency_error': abs(actual_transition_freq - predicted_transition_freq),
            'n_actual_transitions': actual_transitions.sum(),
            'n_predicted_transitions': predicted_transitions.sum()
        }
    
    def _analyze_cloud_fluctuations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze cloud-induced power fluctuations"""
        
        # Calculate power changes (derivative)
        df['actual_power_change'] = df['actual'].diff().abs()
        df['predicted_power_change'] = df['predicted'].diff().abs()
        
        # Remove NaN values
        df_clean = df.dropna()
        
        if len(df_clean) < 10:
            return {'correlation': 0, 'mean_error': 0, 'std_error': 0}
        
        # Calculate fluctuation correlation
        fluctuation_correlation = df_clean['actual_power_change'].corr(df_clean['predicted_power_change'])
        
        # Calculate fluctuation errors
        fluctuation_errors = df_clean['actual_power_change'] - df_clean['predicted_power_change']
        mean_fluctuation_error = fluctuation_errors.mean()
        std_fluctuation_error = fluctuation_errors.std()
        
        # Analyze high-fluctuation periods
        high_fluctuation_threshold = df_clean['actual_power_change'].quantile(0.8)
        high_fluctuation_mask = df_clean['actual_power_change'] > high_fluctuation_threshold
        
        if high_fluctuation_mask.sum() > 0:
            high_fluctuation_correlation = df_clean[high_fluctuation_mask]['actual_power_change'].corr(
                df_clean[high_fluctuation_mask]['predicted_power_change']
            )
            high_fluctuation_error = df_clean[high_fluctuation_mask]['actual_power_change'].mean() - \
                                   df_clean[high_fluctuation_mask]['predicted_power_change'].mean()
        else:
            high_fluctuation_correlation = 0
            high_fluctuation_error = 0
        
        return {
            'fluctuation_correlation': fluctuation_correlation,
            'mean_fluctuation_error': mean_fluctuation_error,
            'std_fluctuation_error': std_fluctuation_error,
            'high_fluctuation_correlation': high_fluctuation_correlation,
            'high_fluctuation_error': high_fluctuation_error,
            'n_high_fluctuation_points': high_fluctuation_mask.sum()
        }
    
    def _calculate_intermittency_prediction_quality(self, daily_intermittency: pd.DataFrame, 
                                                  intermittency_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall prediction quality for intermittency characteristics"""
        
        if len(daily_intermittency) == 0:
            return {'mean_intermittency_error': 0, 'transition_accuracy': 0}
        
        # Calculate mean errors
        mean_intermittency_error = daily_intermittency['intermittency_ratio_error'].mean()
        mean_on_duration_error = daily_intermittency['on_duration_error'].mean()
        mean_off_duration_error = daily_intermittency['off_duration_error'].mean()
        mean_transition_error = daily_intermittency['transition_error'].mean()
        
        # Calculate correlations
        intermittency_correlation = daily_intermittency['actual_intermittency_ratio'].corr(
            daily_intermittency['predicted_intermittency_ratio']
        )
        on_duration_correlation = daily_intermittency['actual_on_duration'].corr(
            daily_intermittency['predicted_on_duration']
        )
        off_duration_correlation = daily_intermittency['actual_off_duration'].corr(
            daily_intermittency['predicted_off_duration']
        )
        
        return {
            'mean_intermittency_error': mean_intermittency_error,
            'mean_on_duration_error': mean_on_duration_error,
            'mean_off_duration_error': mean_off_duration_error,
            'mean_transition_error': mean_transition_error,
            'intermittency_correlation': intermittency_correlation,
            'on_duration_correlation': on_duration_correlation,
            'off_duration_correlation': off_duration_correlation,
            'overall_intermittency_score': 1 - mean_intermittency_error  # Higher is better
        }
    
    def _identify_problematic_periods(self, df: pd.DataFrame, 
                                    daily_intermittency: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify periods with poor intermittency prediction"""
        
        if len(daily_intermittency) == 0:
            return []
        
        # Identify days with high intermittency prediction errors
        high_error_threshold = daily_intermittency['intermittency_ratio_error'].quantile(0.75)
        problematic_days = daily_intermittency[
            daily_intermittency['intermittency_ratio_error'] > high_error_threshold
        ]
        
        problematic_periods = []
        for _, day in problematic_days.iterrows():
            day_df = df[df['timestamp'].dt.date == day['date']]
            
            if len(day_df) > 0:
                problematic_periods.append({
                    'date': day['date'],
                    'intermittency_error': day['intermittency_ratio_error'],
                    'on_duration_error': day['on_duration_error'],
                    'off_duration_error': day['off_duration_error'],
                    'transition_error': day['transition_error'],
                    'start_time': day_df['timestamp'].min(),
                    'end_time': day_df['timestamp'].max(),
                    'n_points': len(day_df)
                })
        
        return problematic_periods
    
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
    
    def create_intermittency_visualizations(self, intermittency_results: Dict[str, Any], 
                                          save_path: str = None) -> Dict[str, plt.Figure]:
        """
        Create visualizations for intermittency prediction quality analysis
        
        Args:
            intermittency_results: Results from analyze_intermittency_prediction_quality
            save_path: Optional path to save figures
            
        Returns:
            Dictionary of matplotlib figures
        """
        figures = {}
        
        if not intermittency_results or 'daily_intermittency' not in intermittency_results:
            print("[WARN] No intermittency data available for visualization")
            return figures
        
        daily_df = intermittency_results['daily_intermittency']
        
        # 1. Daily Intermittency Ratio Comparison
        fig1, (ax1a, ax1b) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Intermittency ratio over time
        ax1a.plot(daily_df['date'], daily_df['actual_intermittency_ratio'], 'o-', 
                 label='Actual Intermittency Ratio', alpha=0.7)
        ax1a.plot(daily_df['date'], daily_df['predicted_intermittency_ratio'], 's-', 
                 label='Predicted Intermittency Ratio', alpha=0.7)
        ax1a.set_xlabel('Date')
        ax1a.set_ylabel('Intermittency Ratio')
        ax1a.set_title(f'Daily Intermittency Ratio Prediction - {self.module_type.title()} Modules')
        ax1a.legend()
        ax1a.grid(True, alpha=0.3)
        
        # Intermittency ratio errors over time
        ax1b.plot(daily_df['date'], daily_df['intermittency_ratio_error'], 'ro-', alpha=0.7)
        ax1b.set_xlabel('Date')
        ax1b.set_ylabel('Intermittency Ratio Error')
        ax1b.set_title(f'Daily Intermittency Ratio Prediction Errors - {self.module_type.title()} Modules')
        ax1b.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        figures['daily_intermittency'] = fig1
        
        # 2. State Duration Analysis
        fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(15, 6))
        
        # On duration comparison
        ax2a.scatter(daily_df['actual_on_duration'], daily_df['predicted_on_duration'], alpha=0.7)
        min_val = min(daily_df['actual_on_duration'].min(), daily_df['predicted_on_duration'].min())
        max_val = max(daily_df['actual_on_duration'].max(), daily_df['predicted_on_duration'].max())
        ax2a.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        ax2a.set_xlabel('Actual On Duration')
        ax2a.set_ylabel('Predicted On Duration')
        ax2a.set_title(f'On Duration Prediction - {self.module_type.title()}')
        ax2a.legend()
        ax2a.grid(True, alpha=0.3)
        corr_on = daily_df['actual_on_duration'].corr(daily_df['predicted_on_duration'])
        ax2a.text(0.05, 0.95, f'R = {corr_on:.3f}', transform=ax2a.transAxes, 
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Off duration comparison
        ax2b.scatter(daily_df['actual_off_duration'], daily_df['predicted_off_duration'], alpha=0.7)
        min_val = min(daily_df['actual_off_duration'].min(), daily_df['predicted_off_duration'].min())
        max_val = max(daily_df['actual_off_duration'].max(), daily_df['predicted_off_duration'].max())
        ax2b.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        ax2b.set_xlabel('Actual Off Duration')
        ax2b.set_ylabel('Predicted Off Duration')
        ax2b.set_title(f'Off Duration Prediction - {self.module_type.title()}')
        ax2b.legend()
        ax2b.grid(True, alpha=0.3)
        corr_off = daily_df['actual_off_duration'].corr(daily_df['predicted_off_duration'])
        ax2b.text(0.05, 0.95, f'R = {corr_off:.3f}', transform=ax2b.transAxes, 
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        figures['state_duration_analysis'] = fig2
        
        # 3. Error Distribution Analysis
        fig3, (ax3a, ax3b, ax3c) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Intermittency ratio error distribution
        ax3a.hist(daily_df['intermittency_ratio_error'], bins=20, alpha=0.7, edgecolor='black')
        ax3a.set_xlabel('Intermittency Ratio Error')
        ax3a.set_ylabel('Frequency')
        ax3a.set_title(f'Intermittency Ratio Error Distribution - {self.module_type.title()}')
        ax3a.grid(True, alpha=0.3)
        
        # On duration error distribution
        ax3b.hist(daily_df['on_duration_error'], bins=20, alpha=0.7, edgecolor='black')
        ax3b.set_xlabel('On Duration Error')
        ax3b.set_ylabel('Frequency')
        ax3b.set_title(f'On Duration Error Distribution - {self.module_type.title()}')
        ax3b.grid(True, alpha=0.3)
        
        # Transition error distribution
        ax3c.hist(daily_df['transition_error'], bins=20, alpha=0.7, edgecolor='black')
        ax3c.set_xlabel('Transition Error')
        ax3c.set_ylabel('Frequency')
        ax3c.set_title(f'Transition Error Distribution - {self.module_type.title()}')
        ax3c.grid(True, alpha=0.3)
        
        figures['error_distributions'] = fig3
        
        # Save figures if path provided
        if save_path:
            for name, fig in figures.items():
                fig.savefig(f"{save_path}/intermittency_analysis_{name}_{self.module_type}.png", 
                           dpi=300, bbox_inches='tight')
        
        return figures


def analyze_intermittency_prediction_quality(predictions: np.ndarray, 
                                           actuals: np.ndarray, 
                                           timestamps: pd.DatetimeIndex,
                                           module_type: str = 'silicon') -> Dict[str, Any]:
    """
    Convenience function to analyze intermittency prediction quality
    
    Args:
        predictions: Model predictions
        actuals: Actual values  
        timestamps: Corresponding timestamps
        module_type: 'silicon' or 'perovskite'
        
    Returns:
        Intermittency prediction quality analysis results
    """
    analyzer = IntermittencyAnalysis(module_type)
    return analyzer.analyze_intermittency_prediction_quality(predictions, actuals, timestamps)


if __name__ == "__main__":
    # Example usage
    print("Analytical Package 3: Intermittency Analysis")
    print("This package analyzes intermittency prediction quality for PV forecasting models")
