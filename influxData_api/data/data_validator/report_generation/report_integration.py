"""
Report Integration Helper
========================

This module provides integration functions to add report generation
to existing validation modes without major code restructuring.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

from .validation_report_generator import ValidationReportGenerator

logger = logging.getLogger(__name__)


class ValidationStepTracker:
    """
    Helper class to track validation steps and integrate with ValidationReportGenerator.
    
    This class can be easily integrated into existing validation modes.
    """
    
    def __init__(self, dataset_name: str, output_dir: Path):
        """
        Initialize the step tracker.
        
        Args:
            dataset_name: Name of the dataset being validated
            output_dir: Directory to save reports
        """
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.reporter = ValidationReportGenerator(output_dir, dataset_name)
        self.current_data = None
        
    def track_step(self, step_name: str, 
                   input_data: pd.DataFrame, 
                   output_data: pd.DataFrame,
                   description: str = None,
                   details: str = None,
                   custom_stats: Dict[str, Any] = None) -> None:
        """
        Track a validation step.
        
        Args:
            step_name: Name of the validation step
            input_data: Data before the step
            output_data: Data after the step
            description: Brief description of what the step does
            details: Additional details about the step
            custom_stats: Custom statistics to include
        """
        input_count = len(input_data) if input_data is not None else 0
        output_count = len(output_data) if output_data is not None else 0
        removed_count = input_count - output_count
        
        # Create data sample for visualization (limit size for performance)
        data_sample = None
        if output_data is not None and len(output_data) > 0:
            sample_size = min(5000, len(output_data))
            if sample_size < len(output_data):
                data_sample = output_data.sample(n=sample_size, random_state=42)
            else:
                data_sample = output_data.copy()
        
        step_data = {
            'description': description or f"Applied {step_name} validation",
            'details': details,
            'input_records': input_count,
            'output_records': output_count,
            'removed_records': removed_count,
            'data_sample': data_sample,
            'statistics': custom_stats or {}
        }
        
        # Add data quality metrics if possible
        if output_data is not None and len(output_data) > 0:
            step_data['statistics'].update(self._calculate_data_quality_metrics(output_data))
        
        self.reporter.add_validation_step(step_name, step_data)
        logger.info(f"Tracked validation step: {step_name} ({input_count} → {output_count} records)")
    
    def track_outlier_detection(self, step_name: str, 
                               input_data: pd.DataFrame,
                               output_data: pd.DataFrame,
                               outliers_detected: Dict[str, List] = None,
                               detection_methods: List[str] = None) -> None:
        """
        Track outlier detection step with specific outlier statistics.
        
        Args:
            step_name: Name of the outlier detection step
            input_data: Data before outlier detection
            output_data: Data after outlier removal
            outliers_detected: Dictionary of outliers by column/method
            detection_methods: List of detection methods used
        """
        custom_stats = {}
        
        if outliers_detected:
            for column, outliers in outliers_detected.items():
                custom_stats[f"Outliers in {column}"] = len(outliers) if outliers else 0
        
        if detection_methods:
            custom_stats["Detection Methods"] = ", ".join(detection_methods)
        
        details = "Outlier detection using statistical methods to identify and remove anomalous data points."
        if detection_methods:
            details += f" Methods used: {', '.join(detection_methods)}"
        
        self.track_step(
            step_name, 
            input_data, 
            output_data,
            description="Statistical outlier detection and removal",
            details=details,
            custom_stats=custom_stats
        )
    
    def track_correlation_validation(self, step_name: str,
                                   input_data: pd.DataFrame,
                                   output_data: pd.DataFrame,
                                   anomalies_detected: List[Dict] = None) -> None:
        """
        Track correlation validation step.
        
        Args:
            step_name: Name of the correlation validation step
            input_data: Data before validation
            output_data: Data after anomaly removal
            anomalies_detected: List of detected correlation anomalies
        """
        custom_stats = {}
        
        if anomalies_detected:
            custom_stats["Correlation Anomalies"] = len(anomalies_detected)
            
            # Analyze anomaly types
            if anomalies_detected:
                avg_irr_change = np.mean([abs(a.get('irr_change_pct', 0)) for a in anomalies_detected])
                avg_p_change = np.mean([abs(a.get('p_change_pct', 0)) for a in anomalies_detected])
                custom_stats["Avg Irr Change (%)"] = round(avg_irr_change, 2)
                custom_stats["Avg P Change (%)"] = round(avg_p_change, 2)
        
        details = "Validation of Irradiance-Power correlation to detect unrealistic changes in power output relative to irradiance changes."
        
        self.track_step(
            step_name,
            input_data,
            output_data,
            description="Irradiance-Power correlation validation",
            details=details,
            custom_stats=custom_stats
        )
    
    def track_aggregation_step(self, step_name: str,
                              input_data: pd.DataFrame,
                              output_data: pd.DataFrame,
                              aggregation_method: str = "average",
                              grouping_columns: List[str] = None) -> None:
        """
        Track data aggregation step.
        
        Args:
            step_name: Name of the aggregation step
            input_data: Data before aggregation
            output_data: Data after aggregation
            aggregation_method: Method used for aggregation
            grouping_columns: Columns used for grouping
        """
        custom_stats = {
            "Aggregation Method": aggregation_method,
            "Compression Ratio": f"{len(input_data)/len(output_data):.2f}:1" if len(output_data) > 0 else "N/A"
        }
        
        if grouping_columns:
            custom_stats["Grouping Columns"] = ", ".join(grouping_columns)
        
        # Calculate unique timestamps
        if '_time' in input_data.columns:
            unique_timestamps = input_data['_time'].nunique()
            custom_stats["Unique Timestamps"] = unique_timestamps
        
        details = f"Data aggregation using {aggregation_method} method to combine multiple records."
        if grouping_columns:
            details += f" Grouped by: {', '.join(grouping_columns)}"
        
        self.track_step(
            step_name,
            input_data,
            output_data,
            description=f"Data aggregation ({aggregation_method})",
            details=details,
            custom_stats=custom_stats
        )
    
    def _calculate_data_quality_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate basic data quality metrics.
        
        Args:
            data: DataFrame to analyze
            
        Returns:
            Dictionary of data quality metrics
        """
        metrics = {}
        
        if len(data) == 0:
            return metrics
        
        # Basic metrics
        metrics["Total Rows"] = len(data)
        metrics["Total Columns"] = len(data.columns)
        
        # Missing data analysis
        missing_data = data.isnull().sum().sum()
        total_cells = len(data) * len(data.columns)
        metrics["Missing Values"] = missing_data
        metrics["Data Completeness (%)"] = round((1 - missing_data/total_cells) * 100, 2) if total_cells > 0 else 0
        
        # Numerical columns analysis
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            metrics["Numeric Columns"] = len(numeric_columns)
            
            # Check for common PV data columns
            if 'P' in data.columns:
                metrics["Avg Power (W)"] = round(data['P'].mean(), 2)
                metrics["Max Power (W)"] = round(data['P'].max(), 2)
            
            if 'Irr' in data.columns:
                metrics["Avg Irradiance (W/m²)"] = round(data['Irr'].mean(), 2)
                metrics["Max Irradiance (W/m²)"] = round(data['Irr'].max(), 2)
        
        # Time range analysis
        if '_time' in data.columns:
            time_range = data['_time'].max() - data['_time'].min()
            metrics["Time Range (days)"] = time_range.days
        
        return metrics
    
    def track_range_validation(self, input_records: int, output_records: int, 
                               removed_records: int, violations: List = None, 
                               data_sample: pd.DataFrame = None) -> None:
        """Track range validation step."""
        step_data = {
            'description': 'Range validation checks for out-of-bounds values',
            'input_records': input_records,
            'output_records': output_records,
            'removed_records': removed_records,
            'statistics': {
                'violations_detected': len(violations) if violations else 0,
                'validation_success_rate': (output_records / input_records) * 100 if input_records > 0 else 0
            },
            'data_sample': data_sample
        }
        self.reporter.add_validation_step("Range Validation", step_data)
    
    def track_quantile_scaling(self, input_records: int, output_records: int,
                               scaling_factors: Dict = None, p_normalized_generated: bool = False,
                               data_sample: pd.DataFrame = None) -> None:
        """Track quantile scaling step."""
        step_data = {
            'description': 'Quantile-based data normalization (P_normalized generation)',
            'input_records': input_records,
            'output_records': output_records,
            'removed_records': 0,  # Scaling doesn't remove records
            'statistics': {
                'scaling_method': 'Quantile-based (0-1.2 range)',
                'p_normalized_generated': 'Yes' if p_normalized_generated else 'No',
                'scaling_factors': len(scaling_factors) if scaling_factors else 0
            },
            'data_sample': data_sample
        }
        self.reporter.add_validation_step("Quantile Scaling Generation", step_data)
    
    def track_cross_module_irr(self, input_records: int, output_records: int,
                               irr_si_added: int = 0, irr_pvk_added: int = 0,
                               data_sample: pd.DataFrame = None) -> None:
        """Track cross-module Irr addition step."""
        step_data = {
            'description': 'Addition of cross-module irradiance values (Irr_si, Irr_pvk)',
            'input_records': input_records,
            'output_records': output_records,
            'removed_records': 0,  # This step adds columns, doesn't remove records
            'statistics': {
                'irr_si_added': irr_si_added,
                'irr_pvk_added': irr_pvk_added,
                'cross_module_enhancement': 'Complete' if output_records > 0 else 'Failed'
            },
            'irr_si_added': irr_si_added,
            'irr_pvk_added': irr_pvk_added,
            'data_sample': data_sample
        }
        self.reporter.add_validation_step("Add Cross-Module Irr Values", step_data)
    
    def track_irr_gap_filling(self, input_records: int, output_records: int,
                              missing_values: int = 0, filled_values: int = 0,
                              data_sample: pd.DataFrame = None) -> None:
        """Track Irr gap filling step."""
        step_data = {
            'description': 'Filling missing irradiance values using GS_10 data and scaling factors',
            'input_records': input_records,
            'output_records': output_records,
            'removed_records': 0,  # Gap filling fills values, doesn't remove records
            'statistics': {
                'missing_values': missing_values,
                'filled_values': filled_values,
                'gap_filling_success_rate': (filled_values / missing_values) * 100 if missing_values > 0 else 100,
                'data_completeness_improvement': (output_records / input_records) * 100 if input_records > 0 else 100
            },
            'filled_values': filled_values,
            'missing_values': missing_values,
            'data_sample': data_sample
        }
        self.reporter.add_validation_step("Filling Missing Irr Values", step_data)
    
    def track_module_failure_detection(self, input_records: int, output_records: int,
                                       removed_records: int, failures_detected: int = 0,
                                       data_sample: pd.DataFrame = None) -> None:
        """Track module failure detection step."""
        step_data = {
            'description': 'Detection of underperforming or faulty PV modules',
            'input_records': input_records,
            'output_records': output_records,
            'removed_records': removed_records,
            'statistics': {
                'module_failures_detected': failures_detected,
                'failure_detection_rate': (removed_records / input_records) * 100 if input_records > 0 else 0,
                'module_performance_validation': (output_records / input_records) * 100 if input_records > 0 else 0
            },
            'data_sample': data_sample
        }
        self.reporter.add_validation_step("Module Failure Detection", step_data)
    
    def track_correlation_validation_new(self, input_records: int, output_records: int,
                                        removed_records: int, anomalies_detected: int = 0,
                                        data_sample: pd.DataFrame = None, step_name: str = None) -> None:
        """Track correlation validation step with new interface."""
        step_data = {
            'description': 'Validation of Irradiance-Power correlation to detect unrealistic changes',
            'input_records': input_records,
            'output_records': output_records,
            'removed_records': removed_records,
            'statistics': {
                'anomalies_detected': anomalies_detected,
                'correlation_anomaly_rate': (removed_records / input_records) * 100 if input_records > 0 else 0,
                'data_consistency_improvement': (output_records / input_records) * 100 if input_records > 0 else 0
            },
            'data_sample': data_sample
        }
        
        final_step_name = step_name if step_name else "Correlation Validation (Irr-P Relationship)"
        self.reporter.add_validation_step(final_step_name, step_data)

    def add_summary_statistics(self, stats: Dict[str, Any]) -> None:
        """
        Add summary statistics for the entire validation process.
        
        Args:
            stats: Summary statistics dictionary
        """
        self.reporter.add_summary_statistics(stats)
    
    def generate_report(self) -> str:
        """
        Generate the final validation report.
        
        Returns:
            Path to the generated report
        """
        return self.reporter.generate_report()


def create_step_tracker(dataset_name: str, output_dir: Path) -> ValidationStepTracker:
    """
    Convenience function to create a validation step tracker.
    
    Args:
        dataset_name: Name of the dataset
        output_dir: Output directory for reports
        
    Returns:
        ValidationStepTracker instance
    """
    return ValidationStepTracker(dataset_name, output_dir)
