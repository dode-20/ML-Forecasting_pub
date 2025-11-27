#!/usr/bin/env python3
"""
Data Validator Report Generator
===============================

This module generates comprehensive HTML reports for data validation processes.
It tracks and visualizes each validation step with detailed statistics and plots.

Features:
- Step-by-step validation tracking
- Interactive visualizations with Plotly
- Detailed statistics tables
- Before/after comparisons
- Outlier and anomaly visualizations
- Data quality metrics
- Professional HTML report generation

Usage:
    from report_generation import ValidationReportGenerator
    
    reporter = ValidationReportGenerator(output_dir="validation_results")
    reporter.add_validation_step("Range Validation", {...})
    reporter.generate_report()
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from plotly.io import to_html
import logging

logger = logging.getLogger(__name__)


class ValidationReportGenerator:
    """
    Comprehensive report generator for data validation processes.
    
    This class tracks validation steps and generates detailed HTML reports
    with visualizations and statistics for each step.
    """
    
    def __init__(self, output_dir: Union[str, Path], dataset_name: str = None):
        """
        Initialize the validation report generator.
        
        Args:
            output_dir: Directory to save the reports
            dataset_name: Name of the dataset being validated
        """
        self.output_dir = Path(output_dir)
        self.dataset_name = dataset_name or "Unknown Dataset"
        self.validation_steps = []
        self.summary_stats = {}
        self.start_time = datetime.now()
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize counters
        self.step_counter = 0
        
        logger.info(f"ValidationReportGenerator initialized for {self.dataset_name}")
    
    def add_validation_step(self, step_name: str, step_data: Dict[str, Any]):
        """
        Add a validation step to the report.
        
        Args:
            step_name: Name of the validation step
            step_data: Dictionary containing step results and statistics
            
        Expected step_data structure:
        {
            'description': 'Brief description of the step',
            'input_records': int,
            'output_records': int,
            'removed_records': int,
            'removed_details': {...},  # Optional: Details about removed records
            'statistics': {...},       # Optional: Statistical summaries
            'data_sample': pd.DataFrame,  # Optional: Sample of data for visualization
            'plots': {...}            # Optional: Custom plot data
        }
        """
        self.step_counter += 1
        
        step_info = {
            'step_number': self.step_counter,
            'step_name': step_name,
            'timestamp': datetime.now(),
            'data': step_data.copy()
        }
        
        self.validation_steps.append(step_info)
        logger.info(f"Added validation step {self.step_counter}: {step_name}")
    
    def add_summary_statistics(self, stats: Dict[str, Any]):
        """
        Add overall summary statistics for the validation process.
        
        Args:
            stats: Dictionary containing summary statistics
        """
        self.summary_stats.update(stats)
    
    
    def create_validation_step_table(self, step_info: Dict[str, Any]) -> str:
        """
        Create a clean table for a specific validation step, focusing on research-relevant metrics.
        
        Args:
            step_info: Step information dictionary
            
        Returns:
            HTML string containing the validation step table
        """
        step_data = step_info['data']
        step_name = step_info['step_name']
        
        # Define step-specific metrics based on validation step type
        if "Range validation" in step_name:
            metrics = {
                'Input Records': f"{step_data.get('input_records', 0):,}",
                'Output Records': f"{step_data.get('output_records', 0):,}",
                'Out-of-Range Records Removed': f"{step_data.get('removed_records', 0):,}",
                'Data Quality Improvement': f"{((step_data.get('removed_records', 0) / step_data.get('input_records', 1)) * 100):.2f}%" if step_data.get('input_records', 0) > 0 else "0%",
                'Range Validation Success Rate': f"{((step_data.get('output_records', 0) / step_data.get('input_records', 1)) * 100):.2f}%" if step_data.get('input_records', 0) > 0 else "0%"
            }
        
        elif "Correlation validation" in step_name:
            metrics = {
                'Input Records': f"{step_data.get('input_records', 0):,}",
                'Output Records': f"{step_data.get('output_records', 0):,}",
                'Irr-P Anomalies Detected': f"{step_data.get('removed_records', 0):,}",
                'Correlation Anomaly Rate': f"{((step_data.get('removed_records', 0) / step_data.get('input_records', 1)) * 100):.2f}%" if step_data.get('input_records', 0) > 0 else "0%",
                'Data Consistency Improvement': f"{((step_data.get('output_records', 0) / step_data.get('input_records', 1)) * 100):.2f}%" if step_data.get('input_records', 0) > 0 else "0%"
            }
        
        elif "Outlier detection" in step_name:
            metrics = {
                'Input Records': f"{step_data.get('input_records', 0):,}",
                'Output Records': f"{step_data.get('output_records', 0):,}",
                'Statistical Outliers Removed': f"{step_data.get('removed_records', 0):,}",
                'Outlier Detection Rate': f"{((step_data.get('removed_records', 0) / step_data.get('input_records', 1)) * 100):.2f}%" if step_data.get('input_records', 0) > 0 else "0%",
                'Data Reliability Improvement': f"{((step_data.get('output_records', 0) / step_data.get('input_records', 1)) * 100):.2f}%" if step_data.get('input_records', 0) > 0 else "0%"
            }
        
        elif "Module failure detection" in step_name:
            metrics = {
                'Input Records': f"{step_data.get('input_records', 0):,}",
                'Output Records': f"{step_data.get('output_records', 0):,}",
                'Module Failures Detected': f"{step_data.get('removed_records', 0):,}",
                'Module Failure Rate': f"{((step_data.get('removed_records', 0) / step_data.get('input_records', 1)) * 100):.2f}%" if step_data.get('input_records', 0) > 0 else "0%",
                'Module Performance Validation': f"{((step_data.get('output_records', 0) / step_data.get('input_records', 1)) * 100):.2f}%" if step_data.get('input_records', 0) > 0 else "0%"
            }
        
        elif "Quantile scaling" in step_name:
            metrics = {
                'Input Records': f"{step_data.get('input_records', 0):,}",
                'Records Processed': f"{step_data.get('output_records', 0):,}",
                'P_normalized Column Generated': 'Yes' if step_data.get('output_records', 0) > 0 else 'No',
                'Scaling Method': 'Quantile-based (0-1.2 range)',
                'Normalization Success Rate': '100%' if step_data.get('output_records', 0) == step_data.get('input_records', 0) else 'Partial'
            }
        
        elif "averages" in step_name.lower() or "aggregat" in step_name.lower():
            metrics = {
                'Input Records (Individual Modules)': f"{step_data.get('input_records', 0):,}",
                'Output Records (Module-Type Averages)': f"{step_data.get('output_records', 0):,}",
                'Data Reduction Ratio': f"{(step_data.get('input_records', 0) / step_data.get('output_records', 1)):.1f}:1" if step_data.get('output_records', 0) > 0 else "N/A",
                'Aggregation Method': 'Validated Module Averaging',
                'Temporal Resolution': '5-minute intervals'
            }
        
        elif "cross-module" in step_name.lower() or "Irr_si" in step_name or "Irr_pvk" in step_name:
            metrics = {
                'Input Records': f"{step_data.get('input_records', 0):,}",
                'Records with Cross-Module Irr': f"{step_data.get('output_records', 0):,}",
                'Irr_si Values Added': f"{step_data.get('irr_si_added', 0):,}" if 'irr_si_added' in step_data else "0",
                'Irr_pvk Values Added': f"{step_data.get('irr_pvk_added', 0):,}" if 'irr_pvk_added' in step_data else "0",
                'Cross-Module Data Enhancement': 'Complete' if step_data.get('output_records', 0) > 0 else 'Failed'
            }
        
        elif "gap filling" in step_name.lower() or "missing" in step_name.lower():
            metrics = {
                'Input Records': f"{step_data.get('input_records', 0):,}",
                'Output Records': f"{step_data.get('output_records', 0):,}",
                'Missing Irr Values Filled': f"{step_data.get('filled_values', 0):,}" if 'filled_values' in step_data else f"{step_data.get('removed_records', 0):,}",
                'Gap Filling Success Rate': f"{((step_data.get('filled_values', 0) / step_data.get('missing_values', 1)) * 100):.2f}%" if step_data.get('missing_values', 0) > 0 else "100%",
                'Data Completeness Improvement': f"{((step_data.get('output_records', 0) / step_data.get('input_records', 1)) * 100):.2f}%" if step_data.get('input_records', 0) > 0 else "100%"
            }
        
        else:
            # Generic metrics for unknown step types
            metrics = {
                'Input Records': f"{step_data.get('input_records', 0):,}",
                'Output Records': f"{step_data.get('output_records', 0):,}",
                'Records Modified': f"{step_data.get('removed_records', 0):,}",
                'Processing Success Rate': f"{((step_data.get('output_records', 0) / step_data.get('input_records', 1)) * 100):.2f}%" if step_data.get('input_records', 0) > 0 else "100%"
            }
        
        # Add any custom statistics from step_data
        if 'statistics' in step_data:
            for key, value in step_data['statistics'].items():
                if key not in metrics:  # Don't override predefined metrics
                    if isinstance(value, (int, float)):
                        if isinstance(value, float):
                            metrics[key] = f"{value:.4f}"
                        else:
                            metrics[key] = f"{value:,}"
                    else:
                        metrics[key] = str(value)
        
        # Create HTML table
        table_rows = ""
        for metric, value in metrics.items():
            table_rows += f"""
                <tr>
                    <td style="font-weight: bold; background-color: #f8f9fa;">{metric}</td>
                    <td>{value}</td>
                </tr>
            """
        
        table_html = f"""
            <table class="metrics-table" style="width: 100%; border-collapse: collapse; margin: 20px 0;">
                <thead>
                    <tr style="background-color: #3498db; color: white;">
                        <th style="padding: 12px; text-align: left; border: 1px solid #ddd;">Metric</th>
                        <th style="padding: 12px; text-align: left; border: 1px solid #ddd;">Value</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
        """
        
        return table_html
    
    def generate_report(self) -> str:
        """
        Generate the complete HTML validation report with forecast_evaluator.py style.
        
        Returns:
            Path to the generated HTML report
        """
        report_path = self.output_dir / f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        # Calculate overall statistics
        if self.validation_steps:
            initial_records = self.validation_steps[0]['data'].get('input_records', 0)
            final_records = self.validation_steps[-1]['data'].get('output_records', 0)
            total_removed = initial_records - final_records
            overall_retention_rate = (final_records / initial_records * 100) if initial_records > 0 else 0
        else:
            initial_records = final_records = total_removed = 0
            overall_retention_rate = 0
        
        # Create overall summary table
        summary_metrics = {
            'Dataset Name': self.dataset_name,
            'Validation Method': 'Comprehensive PV Data Validation',
            'Total Validation Steps': len(self.validation_steps),
            'Initial Records': f"{initial_records:,}",
            'Final Records': f"{final_records:,}",
            'Total Records Removed/Modified': f"{total_removed:,}",
            'Data Retention Rate': f"{overall_retention_rate:.2f}%",
            'Processing Time': f"{(datetime.now() - self.start_time).total_seconds():.1f} seconds",
            'Validation Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        summary_table_rows = ""
        for metric, value in summary_metrics.items():
            summary_table_rows += f"""
                <tr>
                    <td style="font-weight: bold; background-color: #f8f9fa; padding: 8px; border: 1px solid #ddd;">{metric}</td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{value}</td>
                </tr>
            """
        
        summary_table = f"""
            <table style="width: 100%; border-collapse: collapse; margin: 20px 0; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.04);">
                <thead>
                    <tr style="background-color: #3498db; color: white;">
                        <th style="padding: 12px; text-align: left; border: 1px solid #ddd;">Parameter</th>
                        <th style="padding: 12px; text-align: left; border: 1px solid #ddd;">Value</th>
                    </tr>
                </thead>
                <tbody>
                    {summary_table_rows}
                </tbody>
            </table>
        """
        
        # Build HTML report (forecast_evaluator.py style)
        html_content = f"""
        <html>
        <head>
            <title>Data Validation Report - {self.dataset_name}</title>
            <meta name='viewport' content='width=device-width, initial-scale=1.0'>
            <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
            <style>
                body {{
                    font-family: 'Inter', Arial, sans-serif;
                    margin: 40px;
                    background-color: #f5f5f5;
                    color: #22223b;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 10px;
                }}
                .metrics-table {{
                    border-collapse: collapse;
                    margin: 20px 0 30px 0;
                    width: 100%;
                    font-size: 12px;
                    background: white;
                    border-radius: 8px;
                    overflow: hidden;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
                }}
                .metrics-table th, .metrics-table td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                .metrics-table th {{
                    background-color: #3498db;
                    color: white;
                    font-weight: bold;
                }}
                .metrics-table tr:nth-child(even) {{
                    background-color: #f2f2f2;
                }}
                .step-section {{
                    margin: 30px 0;
                    padding: 20px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    background: #f8fafc;
                }}
                .summary-box {{
                    background-color: #ecf0f1;
                    padding: 15px;
                    border-radius: 5px;
                    margin: 20px 0;
                }}
                .timestamp {{
                    color: #7f8c8d;
                    font-style: italic;
                    font-size: 0.98rem;
                }}
                .config-section {{
                    margin: 20px 0;
                    padding: 20px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    background: #f0f8ff;
                }}
                @media (max-width: 800px) {{
                    .container {{
                        padding: 10px 2vw;
                    }}
                    .step-section, .summary-box, .config-section {{
                        padding: 10px 2vw;
                    }}
                    .metrics-table {{
                        font-size: 0.95rem;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Data Validation Report</h1>
                <h2>Dataset: {self.dataset_name}</h2>
                <p class="timestamp"><b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

                <div class="config-section">
                    <h2>Validation Summary</h2>
                    {summary_table}
                </div>

                <div class="summary-box">
                    <h3>Validation Process Overview</h3>
                    <p>This report presents the results of a comprehensive data validation process for PV (photovoltaic) training data. 
                    The validation includes range checks, correlation analysis, outlier detection, module failure identification, 
                    data normalization, aggregation, and gap filling procedures.</p>
                    <p><strong>Methodology:</strong> Multi-step validation pipeline ensuring data quality and consistency for machine learning applications.</p>
                </div>
"""

        # Add each validation step as a separate section
        for step_info in self.validation_steps:
            step_table = self.create_validation_step_table(step_info)
            step_data = step_info['data']
            
            html_content += f"""
                <div class="step-section">
                    <h2>Step {step_info['step_number']}: {step_info['step_name']}</h2>
                    <p class="timestamp">Executed: {step_info['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}</p>
                    {f"<p><strong>Description:</strong> {step_data.get('description', 'No description available.')}</p>" if step_data.get('description') else ""}
                    {step_table}
                </div>
            """

        # Close HTML
        html_content += """
            </div>
        </body>
        </html>
        """

        # Save report
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Validation report generated: {report_path}")
        print(f"[INFO] Validation report saved: {report_path}")
        
        return str(report_path)


def create_validation_report(output_dir: Union[str, Path], dataset_name: str, 
                           validation_steps: List[Dict], summary_stats: Dict = None) -> str:
    """
    Convenience function to create a validation report.
    
    Args:
        output_dir: Directory to save the report
        dataset_name: Name of the dataset
        validation_steps: List of validation step dictionaries
        summary_stats: Optional summary statistics
        
    Returns:
        Path to the generated report
    """
    reporter = ValidationReportGenerator(output_dir, dataset_name)
    
    if summary_stats:
        reporter.add_summary_statistics(summary_stats)
    
    for step in validation_steps:
        reporter.add_validation_step(step['name'], step['data'])
    
    return reporter.generate_report()
