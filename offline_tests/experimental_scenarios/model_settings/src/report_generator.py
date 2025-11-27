#!/usr/bin/env python3
"""
Report generation for hyperparameter analysis experiments.

Creates scientific reports with visualizations and statistical analysis
of hyperparameter impact on model performance.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

class HyperparameterReportGenerator:
    """Generates comprehensive reports for hyperparameter analysis"""
    
    def __init__(self, output_dir: Path):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory for saving reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set matplotlib style for scientific plots
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
    
    def generate_parameter_report(self, parameter_results: Dict[str, Any]) -> Path:
        """
        Generate comprehensive report for single parameter analysis.
        
        Args:
            parameter_results: Results from parameter analysis
            
        Returns:
            Path to generated report file
        """
        parameter_name = parameter_results["parameter_name"]
        timestamp = parameter_results["timestamp"]
        
        report_dir = self.output_dir / f"{parameter_name}_{timestamp}"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate visualizations
        self._create_parameter_plots(parameter_results, report_dir)
        
        # Generate HTML report
        html_report_path = self._create_html_report(parameter_results, report_dir)
        
        # Generate statistical analysis
        stats_report = self._generate_statistical_analysis(parameter_results)
        
        # Create main report document
        report_path = self._create_report_document(parameter_results, stats_report, report_dir)
        
        # Save raw data
        self._save_raw_data(parameter_results, report_dir)
        
        return html_report_path  # Return HTML report as primary output
    
    def generate_combined_report(self, combined_results: Dict[str, Any]) -> Path:
        """
        Generate combined report for multi-parameter analysis.
        
        Args:
            combined_results: Combined results from multiple parameter analyses
            
        Returns:
            Path to generated combined report
        """
        timestamp = combined_results["timestamp"]
        evaluation_mode = combined_results["evaluation_mode"]
        
        report_dir = self.output_dir / f"combined_analysis_{timestamp}"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate comparison visualizations
        self._create_comparison_plots(combined_results, report_dir)
        
        # Create combined report document
        report_path = self._create_combined_report_document(combined_results, report_dir)
        
        return report_path
    
    def _create_parameter_plots(self, parameter_results: Dict[str, Any], output_dir: Path):
        """Create visualization plots for parameter analysis"""
        parameter_name = parameter_results["parameter_name"]
        experiments = parameter_results["experiments"]
        evaluation_mode = parameter_results["evaluation_mode"]
        
        successful_experiments = [exp for exp in experiments if exp.get("success", False)]
        
        if not successful_experiments:
            print(f"No successful experiments for {parameter_name}, skipping plots")
            return
        
        # Extract data for plotting
        parameter_values = []
        metrics_data = []
        
        for exp in successful_experiments:
            parameter_values.append(exp["parameter_value"])
            
            if evaluation_mode == "silicon_only":
                metrics = exp.get("evaluation_results", {}).get("metrics", {})
                metrics_data.append(metrics)
            else:
                # Cross-technology mode
                transfer_metrics = exp.get("evaluation_results", {}).get("transfer_metrics", {})
                metrics_data.append(transfer_metrics)
        
        if not metrics_data:
            return
        
        # Create DataFrame for easier plotting
        df_data = []
        for i, (param_val, metrics) in enumerate(zip(parameter_values, metrics_data)):
            for metric_name, metric_value in metrics.items():
                df_data.append({
                    "parameter_value": param_val,
                    "metric_name": metric_name,
                    "metric_value": metric_value
                })
        
        df = pd.DataFrame(df_data)
        
        if df.empty:
            return
        
        # Create plots based on evaluation mode
        if evaluation_mode == "silicon_only":
            self._create_silicon_plots(df, parameter_name, output_dir)
        else:
            self._create_transferability_plots(df, parameter_name, output_dir)
    
    def _create_silicon_plots(self, df: pd.DataFrame, parameter_name: str, output_dir: Path):
        """Create plots for silicon-only analysis"""
        # Main metrics plot
        main_metrics = ["RMSE", "MAE", "R²", "MAPE"]
        available_metrics = [m for m in main_metrics if m in df["metric_name"].values]
        
        if available_metrics:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.flatten()
            
            for i, metric in enumerate(available_metrics[:4]):
                metric_df = df[df["metric_name"] == metric]
                if not metric_df.empty:
                    axes[i].plot(metric_df["parameter_value"], metric_df["metric_value"], 
                               marker='o', linewidth=2, markersize=8)
                    axes[i].set_xlabel(parameter_name)
                    axes[i].set_ylabel(metric)
                    axes[i].set_title(f"{metric} vs {parameter_name}")
                    axes[i].grid(True, alpha=0.3)
            
            # Remove empty subplots
            for i in range(len(available_metrics), 4):
                axes[i].remove()
            
            plt.tight_layout()
            plt.savefig(output_dir / f"{parameter_name}_silicon_metrics.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_transferability_plots(self, df: pd.DataFrame, parameter_name: str, output_dir: Path):
        """Create plots for transferability analysis"""
        # Degradation metrics plot
        degradation_metrics = [col for col in df["metric_name"].unique() if "degradation" in col]
        
        if degradation_metrics:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.flatten()
            
            for i, metric in enumerate(degradation_metrics[:4]):
                metric_df = df[df["metric_name"] == metric]
                if not metric_df.empty:
                    axes[i].plot(metric_df["parameter_value"], metric_df["metric_value"], 
                               marker='o', linewidth=2, markersize=8, color='red')
                    axes[i].set_xlabel(parameter_name)
                    axes[i].set_ylabel(f"{metric} (%)")
                    axes[i].set_title(f"{metric} vs {parameter_name}")
                    axes[i].grid(True, alpha=0.3)
                    axes[i].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            # Remove empty subplots
            for i in range(len(degradation_metrics), 4):
                axes[i].remove()
            
            plt.tight_layout()
            plt.savefig(output_dir / f"{parameter_name}_transferability_degradation.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_comparison_plots(self, combined_results: Dict[str, Any], output_dir: Path):
        """Create comparison plots across multiple parameters"""
        parameter_results = combined_results["parameter_results"]
        evaluation_mode = combined_results["evaluation_mode"]
        
        # Extract best performance for each parameter
        best_configs = []
        
        for param_name, param_data in parameter_results.items():
            if param_data.get("error"):
                continue
                
            summary = param_data.get("summary", {})
            best_config = summary.get("best_configuration")
            
            if best_config:
                best_configs.append({
                    "parameter": param_name,
                    "best_value": best_config["parameter_value"],
                    **best_config["metrics"]
                })
        
        if not best_configs:
            return
        
        df = pd.DataFrame(best_configs)
        
        # Create comparison bar plot
        metrics = ["RMSE", "MAE", "R²", "MAPE"]
        available_metrics = [m for m in metrics if m in df.columns]
        
        if available_metrics:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()
            
            for i, metric in enumerate(available_metrics[:4]):
                axes[i].bar(df["parameter"], df[metric], alpha=0.7)
                axes[i].set_ylabel(metric)
                axes[i].set_title(f"Best {metric} by Parameter")
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].grid(True, alpha=0.3)
            
            # Remove empty subplots
            for i in range(len(available_metrics), 4):
                axes[i].remove()
            
            plt.tight_layout()
            plt.savefig(output_dir / "parameter_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def _generate_statistical_analysis(self, parameter_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate statistical analysis of parameter impact"""
        experiments = parameter_results["experiments"]
        successful_experiments = [exp for exp in experiments if exp.get("success", False)]
        
        if len(successful_experiments) < 2:
            return {"status": "insufficient_data"}
        
        # Extract performance data
        parameter_values = [exp["parameter_value"] for exp in successful_experiments]
        
        stats = {
            "parameter_name": parameter_results["parameter_name"],
            "n_experiments": len(successful_experiments),
            "parameter_range": {
                "min": min(parameter_values),
                "max": max(parameter_values),
                "values": sorted(list(set(parameter_values)))
            }
        }
        
        return stats
    
    def _create_report_document(self, parameter_results: Dict[str, Any], stats_report: Dict[str, Any], output_dir: Path) -> Path:
        """Create main report document"""
        parameter_name = parameter_results["parameter_name"]
        timestamp = parameter_results["timestamp"]
        
        report_path = output_dir / f"{parameter_name}_analysis_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# Hyperparameter Analysis Report: {parameter_name}\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Evaluation Mode:** {parameter_results['evaluation_mode']}\n")
            f.write(f"**Parameter:** {parameter_name}\n\n")
            
            # Parameter description
            param_config = parameter_results.get("parameter_config", {})
            f.write(f"## Parameter Description\n\n")
            f.write(f"**Description:** {param_config.get('description', 'N/A')}\n")
            f.write(f"**Type:** {param_config.get('type', 'N/A')}\n")
            f.write(f"**Tested Values:** {param_config.get('values', [])}\n\n")
            
            # Experiment summary
            summary = parameter_results.get("summary", {})
            f.write(f"## Experiment Summary\n\n")
            f.write(f"- **Total Experiments:** {summary.get('total_experiments', 0)}\n")
            f.write(f"- **Successful Experiments:** {summary.get('successful_experiments', 0)}\n")
            
            if summary.get("best_configuration"):
                best = summary["best_configuration"]
                f.write(f"- **Best Configuration:** {parameter_name} = {best['parameter_value']}\n")
                f.write(f"- **Best Metrics:** {best.get('metrics', {})}\n")
            
            f.write(f"\n## Statistical Analysis\n\n")
            f.write(f"```json\n{json.dumps(stats_report, indent=2)}\n```\n\n")
            
            # Visualizations
            f.write(f"## Visualizations\n\n")
            f.write(f"![Metrics Plot]({parameter_name}_silicon_metrics.png)\n\n")
            
            # Raw data
            f.write(f"## Raw Data\n\n")
            f.write(f"Detailed experimental data available in: `{parameter_name}_raw_data.json`\n")
        
        return report_path
    
    def _create_combined_report_document(self, combined_results: Dict[str, Any], output_dir: Path) -> Path:
        """Create combined analysis report document"""
        timestamp = combined_results["timestamp"]
        evaluation_mode = combined_results["evaluation_mode"]
        
        report_path = output_dir / f"combined_analysis_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# Combined Hyperparameter Analysis Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Evaluation Mode:** {evaluation_mode}\n")
            f.write(f"**Parameters Analyzed:** {', '.join(combined_results['parameters'])}\n\n")
            
            # Summary
            summary = combined_results.get("summary", {})
            f.write(f"## Analysis Summary\n\n")
            f.write(f"- **Total Parameters Analyzed:** {summary.get('total_parameters_analyzed', 0)}\n")
            f.write(f"- **Successful Parameter Analyses:** {summary.get('successful_parameters', 0)}\n\n")
            
            # Parameter results
            f.write(f"## Parameter Results\n\n")
            parameter_results = combined_results.get("parameter_results", {})
            
            for param_name, param_data in parameter_results.items():
                if param_data.get("error"):
                    f.write(f"### {param_name} (FAILED)\n")
                    f.write(f"Error: {param_data['error']}\n\n")
                else:
                    f.write(f"### {param_name}\n")
                    summary = param_data.get("summary", {})
                    f.write(f"- Experiments: {summary.get('successful_experiments', 0)}/{summary.get('total_experiments', 0)}\n")
                    
                    best_config = summary.get("best_configuration")
                    if best_config:
                        f.write(f"- Best Value: {best_config['parameter_value']}\n")
                        f.write(f"- Best Metrics: {best_config.get('metrics', {})}\n")
                    f.write(f"\n")
            
            # Comparison visualization
            f.write(f"## Parameter Comparison\n\n")
            f.write(f"![Parameter Comparison](parameter_comparison.png)\n\n")
        
        return report_path
    
    def _save_raw_data(self, parameter_results: Dict[str, Any], output_dir: Path):
        """Save raw experimental data"""
        parameter_name = parameter_results["parameter_name"]
        
        # Save complete results
        raw_data_path = output_dir / f"{parameter_name}_raw_data.json"
        
        # Create serializable version of results
        serializable_results = self._make_serializable(parameter_results)
        
        with open(raw_data_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=4)
    
    def _make_serializable(self, obj):
        """Make object JSON serializable"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (Path,)):
            return str(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif hasattr(obj, 'isoformat'):  # Handle datetime/timestamp objects
            return obj.isoformat()
        elif hasattr(obj, '__str__') and ('Timestamp' in str(type(obj)) or 'datetime' in str(type(obj))):
            return str(obj)
        elif str(type(obj)) in ["<class 'pandas._libs.tslibs.timestamps.Timestamp'>", 
                               "<class 'pandas.core.dtypes.dtypes.DatetimeTZDtype'>"]:
            return str(obj)
        else:
            return obj
    
    def _create_html_report(self, parameter_results: Dict[str, Any], output_dir: Path) -> Path:
        """Create interactive HTML report with Plotly visualizations"""
        parameter_name = parameter_results["parameter_name"]
        timestamp = parameter_results["timestamp"]
        evaluation_mode = parameter_results["evaluation_mode"]
        
        html_path = output_dir / f"{parameter_name}_analysis_report.html"
        
        # Extract data for plotting
        experiments = parameter_results["experiments"]
        successful_experiments = [exp for exp in experiments if exp.get("success", False)]
        
        if not successful_experiments:
            # Create error report
            html_content = self._create_error_html(parameter_name, "No successful experiments")
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            return html_path
        
        # Create interactive plots
        plots_html = self._create_interactive_plots(parameter_results, successful_experiments)
        
        # Generate HTML content
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hyperparameter Analysis: {parameter_name}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 2px solid #e0e0e0;
        }}
        .header h1 {{
            color: #2c3e50;
            margin: 0;
        }}
        .header .subtitle {{
            color: #7f8c8d;
            margin-top: 10px;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        .summary-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .summary-card h3 {{
            margin: 0 0 10px 0;
            font-size: 2em;
        }}
        .summary-card p {{
            margin: 0;
            opacity: 0.9;
        }}
        .plot-container {{
            margin: 30px 0;
            padding: 20px;
            background-color: #fafafa;
            border-radius: 8px;
        }}
        .plot-title {{
            font-size: 1.5em;
            color: #2c3e50;
            margin-bottom: 20px;
            text-align: center;
        }}
        .metrics-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .metrics-table th, .metrics-table td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        .metrics-table th {{
            background-color: #34495e;
            color: white;
        }}
        .metrics-table tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .best-config {{
            background-color: #d5f4e6;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .best-config h3 {{
            color: #27ae60;
            margin-top: 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Hyperparameter Analysis Report</h1>
            <div class="subtitle">
                <strong>Parameter:</strong> {parameter_name} | 
                <strong>Mode:</strong> {evaluation_mode} | 
                <strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </div>
        </div>
        
        {self._create_summary_section(parameter_results)}
        
        {plots_html}
        
        {self._create_metrics_table(successful_experiments)}
        
        {self._create_ranking_table(parameter_results)}
        
        {self._create_best_configuration_section(parameter_results)}
        
        <div class="plot-container">
            <h3>Raw Data</h3>
            <p>Complete experimental data available in: <code>{parameter_name}_raw_data.json</code></p>
        </div>
    </div>
</body>
</html>
"""
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return html_path
    
    def _create_summary_section(self, parameter_results: Dict[str, Any]) -> str:
        """Create summary section HTML"""
        summary = parameter_results.get("summary", {})
        parameter_config = parameter_results.get("parameter_config", {})
        
        total_exp = summary.get("total_experiments", 0)
        successful_exp = summary.get("successful_experiments", 0)
        success_rate = (successful_exp / total_exp * 100) if total_exp > 0 else 0
        
        return f"""
        <div class="summary-grid">
            <div class="summary-card">
                <h3>{total_exp}</h3>
                <p>Total Experiments</p>
            </div>
            <div class="summary-card">
                <h3>{successful_exp}</h3>
                <p>Successful Experiments</p>
            </div>
            <div class="summary-card">
                <h3>{success_rate:.1f}%</h3>
                <p>Success Rate</p>
            </div>
            <div class="summary-card">
                <h3>{len(parameter_config.get('values', []))}</h3>
                <p>Parameter Values Tested</p>
            </div>
        </div>
        """
    
    def _create_interactive_plots(self, parameter_results: Dict[str, Any], successful_experiments: List[Dict]) -> str:
        """Create interactive Plotly plots"""
        parameter_name = parameter_results["parameter_name"]
        evaluation_mode = parameter_results["evaluation_mode"]
        
        # Extract data
        parameter_values = [exp["parameter_value"] for exp in successful_experiments]
        
        if evaluation_mode == "silicon_only":
            return self._create_silicon_interactive_plots(parameter_values, successful_experiments, parameter_name)
        else:
            return self._create_transferability_interactive_plots(parameter_values, successful_experiments, parameter_name)
    
    def _create_silicon_interactive_plots(self, parameter_values: List, experiments: List[Dict], parameter_name: str) -> str:
        """Create interactive plots for silicon-only analysis"""
        # Extract metrics
        metrics_data = {}
        for exp in experiments:
            metrics = exp.get("evaluation_results", {}).get("metrics", {})
            param_val = exp["parameter_value"]
            
            for metric_name, metric_value in metrics.items():
                if metric_name not in metrics_data:
                    metrics_data[metric_name] = {"values": [], "params": []}
                metrics_data[metric_name]["values"].append(metric_value)
                metrics_data[metric_name]["params"].append(param_val)
        
        if not metrics_data:
            return "<div class='plot-container'><p>No metrics data available for plotting.</p></div>"
        
        # Create subplots
        main_metrics = ["RMSE", "MAE", "R²", "MAPE"]
        available_metrics = [m for m in main_metrics if m in metrics_data]
        
        if not available_metrics:
            return "<div class='plot-container'><p>No standard metrics available for plotting.</p></div>"
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=available_metrics[:4],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, metric in enumerate(available_metrics[:4]):
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            data = metrics_data[metric]
            fig.add_trace(
                go.Scatter(
                    x=data["params"],
                    y=data["values"],
                    mode='lines+markers',
                    name=metric,
                    line=dict(color=colors[i], width=3),
                    marker=dict(size=8),
                    showlegend=False
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            height=600,
            title_text=f"{parameter_name} Impact on Model Performance",
            title_x=0.5,
            showlegend=False
        )
        
        # Update axes labels
        for i in range(1, 3):
            for j in range(1, 3):
                fig.update_xaxes(title_text=parameter_name, row=i, col=j)
        
        plot_html = pyo.plot(fig, output_type='div', include_plotlyjs=False)
        
        return f"""
        <div class="plot-container">
            <div class="plot-title">Performance Metrics vs {parameter_name}</div>
            {plot_html}
        </div>
        """
    
    def _create_transferability_interactive_plots(self, parameter_values: List, experiments: List[Dict], parameter_name: str) -> str:
        """Create interactive plots for transferability analysis"""
        # Extract transferability metrics
        transfer_metrics = {}
        for exp in experiments:
            metrics = exp.get("evaluation_results", {}).get("transfer_metrics", {})
            param_val = exp["parameter_value"]
            
            for metric_name, metric_value in metrics.items():
                if metric_name not in transfer_metrics:
                    transfer_metrics[metric_name] = {"values": [], "params": []}
                transfer_metrics[metric_name]["values"].append(metric_value)
                transfer_metrics[metric_name]["params"].append(param_val)
        
        if not transfer_metrics:
            return "<div class='plot-container'><p>No transferability metrics available for plotting.</p></div>"
        
        # Focus on degradation metrics
        degradation_metrics = [m for m in transfer_metrics.keys() if "degradation" in m]
        
        if not degradation_metrics:
            return "<div class='plot-container'><p>No degradation metrics available for plotting.</p></div>"
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=degradation_metrics[:4],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        colors = ['#e74c3c', '#c0392b', '#8e44ad', '#9b59b6']
        
        for i, metric in enumerate(degradation_metrics[:4]):
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            data = transfer_metrics[metric]
            fig.add_trace(
                go.Scatter(
                    x=data["params"],
                    y=data["values"],
                    mode='lines+markers',
                    name=metric,
                    line=dict(color=colors[i], width=3),
                    marker=dict(size=8),
                    showlegend=False
                ),
                row=row, col=col
            )
            
            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5, row=row, col=col)
        
        fig.update_layout(
            height=600,
            title_text=f"{parameter_name} Impact on Transferability (Si → Pvk)",
            title_x=0.5,
            showlegend=False
        )
        
        plot_html = pyo.plot(fig, output_type='div', include_plotlyjs=False)
        
        return f"""
        <div class="plot-container">
            <div class="plot-title">Transferability Degradation vs {parameter_name}</div>
            {plot_html}
        </div>
        """
    
    def _create_metrics_table(self, experiments: List[Dict]) -> str:
        """Create metrics comparison table"""
        if not experiments:
            return ""
        
        # Extract metrics for table
        table_data = []
        for exp in experiments:
            param_val = exp["parameter_value"]
            metrics = exp.get("evaluation_results", {}).get("metrics", {})
            
            row = {"Parameter Value": param_val}
            row.update(metrics)
            table_data.append(row)
        
        if not table_data:
            return ""
        
        # Create HTML table
        df = pd.DataFrame(table_data)
        
        table_html = df.to_html(
            classes="metrics-table",
            index=False,
            escape=False,
            table_id="metrics-table"
        )
        
        return f"""
        <div class="plot-container">
            <h3>Detailed Metrics Comparison</h3>
            {table_html}
        </div>
        """
    
    def _create_ranking_table(self, parameter_results: Dict[str, Any]) -> str:
        """Create ranking table with points system"""
        experiments = parameter_results.get("experiments", [])
        successful_experiments = [exp for exp in experiments if exp.get("success", False)]
        
        if len(successful_experiments) < 2:
            return ""
        
        # Prepare data for ranking
        table_data = []
        for exp in successful_experiments:
            row = {
                "parameter_value": exp["parameter_value"],
                "metrics": exp.get("evaluation_results", {}).get("metrics", {})
            }
            table_data.append(row)
        
        if not table_data:
            return ""
        
        # Create DataFrame for easier manipulation
        df_data = []
        for row in table_data:
            data_row = {"parameter_value": row["parameter_value"]}
            data_row.update(row["metrics"])
            df_data.append(data_row)
        
        df = pd.DataFrame(df_data)
        
        # Debug: Print available columns
        print(f"[DEBUG] Available columns in ranking data: {list(df.columns)}")
        print(f"[DEBUG] DataFrame shape: {df.shape}")
        print(f"[DEBUG] DataFrame head:\n{df.head()}")
        
        # Define metrics and their ranking direction
        # Map actual column names to expected metric names
        metrics_config = {
            "rmse": "asc",              # Lower is better (rank 1 = best)
            "mae": "asc",               # Lower is better
            "mape": "asc",              # Lower is better
            "peak_error": "asc",        # Lower is better
            "r2": "desc",               # Higher is better (rank 1 = best)
            "directional_accuracy": "desc"  # Higher is better
        }
        
        # Create ranking table
        ranking_data = []
        for _, row in df.iterrows():
            ranking_row = {"parameter_value": row["parameter_value"]}
            total_points = 0
            
            print(f"[DEBUG] Processing parameter_value: {row['parameter_value']}")
            
            for metric, direction in metrics_config.items():
                if metric in df.columns:
                    if direction == "asc":
                        # Lower values get better ranks (1 = best)
                        rank = df[metric].rank(method='min', ascending=True)[row.name]
                    else:
                        # Higher values get better ranks (1 = best)
                        rank = df[metric].rank(method='min', ascending=False)[row.name]
                    
                    ranking_row[metric] = int(rank)
                    total_points += int(rank)
                    print(f"[DEBUG]   {metric}: {row[metric]:.4f} -> rank {int(rank)}")
                else:
                    print(f"[DEBUG]   {metric}: NOT FOUND in columns")
            
            ranking_row["Total Points"] = total_points
            ranking_data.append(ranking_row)
            print(f"[DEBUG] Total points for {row['parameter_value']}: {total_points}")
        
        # Sort by total points (lower is better)
        ranking_df = pd.DataFrame(ranking_data)
        ranking_df = ranking_df.sort_values("Total Points")
        
        # Create HTML table
        ranking_html = ranking_df.to_html(
            classes="ranking-table",
            index=False,
            escape=False,
            table_id="ranking-table"
        )
        
        # Add CSS styling
        css_style = """
        <style>
        .ranking-table {
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }
        .ranking-table th, .ranking-table td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: center;
        }
        .ranking-table th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        .ranking-table tr:nth-child(1) {
            background-color: #e8f5e8; /* Light green for best */
        }
        .ranking-table tr:nth-child(2) {
            background-color: #fff3cd; /* Light yellow for second */
        }
        .ranking-table tr:nth-child(3) {
            background-color: #f8d7da; /* Light red for third */
        }
        </style>
        """
        
        return f"""
        {css_style}
        <div class="plot-container">
            <h3>Ranking Table (Lower Points = Better)</h3>
            <p><em>Rank 1 = Best performance for each metric. Total points = sum of all ranks.</em></p>
            {ranking_html}
        </div>
        """
    
    def _create_best_configuration_section(self, parameter_results: Dict[str, Any]) -> str:
        """Create best configuration section"""
        summary = parameter_results.get("summary", {})
        best_config = summary.get("best_configuration")
        
        if not best_config:
            return ""
        
        parameter_name = parameter_results["parameter_name"]
        best_value = best_config["parameter_value"]
        best_metrics = best_config.get("metrics", {})
        
        metrics_html = ""
        for metric_name, metric_value in best_metrics.items():
            metrics_html += f"<li><strong>{metric_name}:</strong> {metric_value:.4f}</li>"
        
        return f"""
        <div class="best-config">
            <h3>Best Configuration</h3>
            <p><strong>{parameter_name}:</strong> {best_value}</p>
            <ul>
                {metrics_html}
            </ul>
        </div>
        """
    
    def _create_error_html(self, parameter_name: str, error_message: str) -> str:
        """Create error HTML report"""
        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Error Report: {parameter_name}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 800px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .error-icon {{
            font-size: 4em;
            color: #e74c3c;
            margin-bottom: 20px;
        }}
        .error-message {{
            color: #2c3e50;
            font-size: 1.2em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="error-icon">[WARN]</div>
        <h1>Analysis Failed</h1>
        <div class="error-message">{error_message}</div>
    </div>
</body>
</html>
"""
