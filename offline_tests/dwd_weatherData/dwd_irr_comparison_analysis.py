#!/usr/bin/env python3
"""
DWD GS_10 vs InfluxDB Irr Comparison Analysis Script

This script compares Global Solar Radiation (GS_10) values from DWD weather data
with Irradiance (Irr) values from InfluxDB to calculate scaling factors and
analyze deviations. The analysis is performed per hour and month to identify
temporal patterns in the relationship between these two data sources.

Author: ML-Forecasting Team
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
from pathlib import Path
import argparse
from typing import Tuple, Dict, List
import logging
import jinja2
import base64
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dwd_irr_comparison.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class DWDIrrComparisonAnalyzer:
    """
    Analyzer class for comparing DWD GS_10 values with InfluxDB Irr values.
    """
    
    def __init__(self, dwd_csv_path: str = None, influx_csv_path: str = None, output_dir: str = "results/irr_comparison"):
        """
        Initialize the analyzer.
        
        Args:
            dwd_csv_path: Path to the DWD weather data CSV file (optional, will be auto-detected)
            influx_csv_path: Path to the InfluxDB data CSV file (optional, will be auto-detected)
            output_dir: Directory to save results and plots
        """
        self.dwd_csv_path = dwd_csv_path
        self.influx_csv_path = influx_csv_path
        
        # Create date-based output directory
        current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir) / current_date
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.dwd_data = None
        self.influx_data = None
        self.combined_data = None
        self.scaling_factors = None
        
        # Analysis results
        self.hourly_factors = None
        self.daily_factors = None
        self.monthly_factors = None
        self.seasonal_factors = None
        
        # Analysis metadata
        self.analysis_metadata = {
            'analysis_date': datetime.now().isoformat(),
            'dwd_csv_path': str(dwd_csv_path),
            'influx_csv_path': str(influx_csv_path),
            'output_directory': str(self.output_dir)
        }
    
    def auto_detect_data_files(self):
        """
        Automatically detect available data files in rawData directory.
        """
        logger.info("Auto-detecting data files...")
        
        # Auto-detect InfluxDB data
        if not self.influx_csv_path:
            raw_data_dir = Path("results/irr_comparison/rawData")
            if raw_data_dir.exists():
                influx_files = list(raw_data_dir.glob("*clean-10min.csv"))
                if influx_files:
                    self.influx_csv_path = str(influx_files[0])
                    logger.info(f"Auto-detected InfluxDB data: {self.influx_csv_path}")
                else:
                    logger.warning("No InfluxDB data files found in rawData directory")
        
        # Auto-detect DWD data
        if not self.dwd_csv_path:
            raw_data_dir = Path("results/irr_comparison/rawData")
            if raw_data_dir.exists():
                dwd_files = list(raw_data_dir.glob("combined_weatherData-10min.csv"))
                if dwd_files:
                    self.dwd_csv_path = str(dwd_files[0])
                    logger.info(f"Auto-detected DWD data: {self.dwd_csv_path}")
                else:
                    logger.warning("No DWD data files found in rawData directory")
        
        # Check if both files are available
        if not self.influx_csv_path or not self.dwd_csv_path:
            missing = []
            if not self.influx_csv_path:
                missing.append("InfluxDB data (*clean-10min.csv)")
            if not self.dwd_csv_path:
                missing.append("DWD data (combined_weatherData-10min.csv)")
            
            raise Exception(f"Missing data files in results/irr_comparison/rawData/: {', '.join(missing)}")
        
        logger.info("Data files auto-detection completed successfully!")
        
    def load_dwd_data(self) -> pd.DataFrame:
        """
        Load and preprocess DWD weather data.
        
        Returns:
            DataFrame with timestamp and GS_10 columns
        """
        logger.info("Loading DWD weather data...")
        
        try:
            # Load DWD data
            dwd_data = pd.read_csv(self.dwd_csv_path)
            
            # Extract timestamp, GS_10, and day_of_year columns
            if 'timestamp' in dwd_data.columns and 'GS_10' in dwd_data.columns:
                if 'day_of_year' in dwd_data.columns:
                    dwd_clean = dwd_data[['timestamp', 'GS_10', 'day_of_year']].copy()
                else:
                    dwd_clean = dwd_data[['timestamp', 'GS_10']].copy()
                    # Calculate day_of_year if not present
                    dwd_clean['timestamp'] = pd.to_datetime(dwd_clean['timestamp'])
                    dwd_clean['day_of_year'] = dwd_clean['timestamp'].dt.dayofyear
            else:
                raise ValueError("Required columns 'timestamp' and 'GS_10' not found in DWD data")
            
            # Convert timestamp to datetime
            dwd_clean['timestamp'] = pd.to_datetime(dwd_clean['timestamp'])
            
            # Remove rows with missing GS_10 values
            dwd_clean = dwd_clean.dropna(subset=['GS_10'])
            
            # Filter out nighttime values (GS_10 = 0) for more meaningful comparison
            dwd_clean = dwd_clean[dwd_clean['GS_10'] > 0]
            
            # Add time components
            dwd_clean['hour'] = dwd_clean['timestamp'].dt.hour
            dwd_clean['month'] = dwd_clean['timestamp'].dt.month
            dwd_clean['season'] = dwd_clean['timestamp'].dt.month.map({
                12: 'Winter', 1: 'Winter', 2: 'Winter',
                3: 'Spring', 4: 'Spring', 5: 'Spring',
                6: 'Summer', 7: 'Summer', 8: 'Summer',
                9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
            })
            
            logger.info(f"Loaded {len(dwd_clean)} DWD records with valid GS_10 values")
            self.dwd_data = dwd_clean
            return dwd_clean
            
        except Exception as e:
            logger.error(f"Error loading DWD data: {e}")
            raise
    
    def load_influx_data(self) -> pd.DataFrame:
        """
        Load and preprocess InfluxDB data.
        
        Returns:
            DataFrame with timestamp and Irr columns
        """
        logger.info("Loading InfluxDB data...")
        
        try:
            # Load InfluxDB data
            influx_data = pd.read_csv(self.influx_csv_path)
            
            # Check for different possible column names for Irr
            irr_column = None
            for col in ['Irr', 'irr', 'IRR', 'irradiance', 'G']:
                if col in influx_data.columns:
                    irr_column = col
                    break
            
            if irr_column is None:
                raise ValueError("No Irr column found in InfluxDB data. Available columns: " + 
                               str(list(influx_data.columns)))
            
            # Check for timestamp column - handle both _time and timestamp
            timestamp_column = None
            for col in ['_time', 'timestamp', 'time', 'Datetime', 'datetime']:
                if col in influx_data.columns:
                    timestamp_column = col
                    break
            
            if timestamp_column is None:
                raise ValueError("No timestamp column found in InfluxDB data. Available columns: " + 
                               str(list(influx_data.columns)))
            
            # Extract timestamp, Irr, and day_of_year columns
            columns_to_extract = [timestamp_column, irr_column]
            if 'day_of_year' in influx_data.columns:
                columns_to_extract.append('day_of_year')
            
            influx_clean = influx_data[columns_to_extract].copy()
            influx_clean.columns = ['timestamp', 'Irr'] + (['day_of_year'] if 'day_of_year' in influx_data.columns else [])
            
            # Convert timestamp to datetime
            influx_clean['timestamp'] = pd.to_datetime(influx_clean['timestamp'])
            
            # Calculate day_of_year if not present
            if 'day_of_year' not in influx_clean.columns:
                influx_clean['day_of_year'] = influx_clean['timestamp'].dt.dayofyear
            
            # Remove rows with missing Irr values
            influx_clean = influx_clean.dropna(subset=['Irr'])
            
            # Filter out nighttime values (Irr = 0) for more meaningful comparison
            influx_clean = influx_clean[influx_clean['Irr'] > 0]
            
            # Add time components
            influx_clean['hour'] = influx_clean['timestamp'].dt.hour
            influx_clean['month'] = influx_clean['timestamp'].dt.month
            influx_clean['season'] = influx_clean['timestamp'].dt.month.map({
                12: 'Winter', 1: 'Winter', 2: 'Winter',
                3: 'Spring', 4: 'Spring', 5: 'Spring',
                6: 'Summer', 7: 'Summer', 8: 'Summer',
                9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
            })
            
            logger.info(f"Loaded {len(influx_clean)} InfluxDB records with valid Irr values")
            self.influx_data = influx_clean
            return influx_clean
            
        except Exception as e:
            logger.error(f"Error loading InfluxDB data: {e}")
            raise
    
    def merge_data(self) -> pd.DataFrame:
        """
        Merge DWD and InfluxDB data based on timestamp matching.
        
        Returns:
            DataFrame with matched DWD and InfluxDB values
        """
        logger.info("Merging DWD and InfluxDB data...")
        
        if self.dwd_data is None or self.influx_data is None:
            raise ValueError("Both DWD and InfluxDB data must be loaded first")
        
        # Round timestamps to nearest minute for better matching
        dwd_rounded = self.dwd_data.copy()
        dwd_rounded['timestamp_rounded'] = dwd_rounded['timestamp'].dt.round('min')
        
        influx_rounded = self.influx_data.copy()
        influx_rounded['timestamp_rounded'] = influx_rounded['timestamp'].dt.round('min')
        
        # Merge on rounded timestamp
        merged = pd.merge(
            dwd_rounded, 
            influx_rounded[['timestamp_rounded', 'Irr']], 
            on='timestamp_rounded', 
            how='inner'
        )
        
        # Clean up
        merged = merged.drop('timestamp_rounded', axis=1)
        
        # Calculate scaling factor
        merged['scaling_factor'] = merged['Irr'] / merged['GS_10']
        
        # Add additional time components
        merged['hour'] = merged['timestamp'].dt.hour
        merged['month'] = merged['timestamp'].dt.month
        merged['season'] = merged['timestamp'].dt.month.map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
        })
        
        # Filter out invalid scaling factors
        merged = merged[
            (merged['scaling_factor'] > 0) & 
            (merged['scaling_factor'] < 10) &  # Reasonable range
            (merged['GS_10'] > 0) & 
            (merged['Irr'] > 0)
        ]
        
        logger.info(f"Merged data contains {len(merged)} matched records")
        self.combined_data = merged
        
        # Save the combined dataframe as CSV
        combined_csv_path = self.output_dir / 'combined_dwd_irr_data.csv'
        merged.to_csv(combined_csv_path, index=False)
        logger.info(f"Combined data saved to {combined_csv_path}")
        
        return merged
    
    def calculate_scaling_factors(self) -> Dict:
        """
        Calculate scaling factors for different time periods.
        
        Returns:
            Dictionary containing various scaling factor calculations
        """
        logger.info("Calculating scaling factors...")
        
        if self.combined_data is None:
            raise ValueError("Data must be merged first")
        
        # Add daytime classification to combined data
        def classify_daytime(hour):
            if 5 <= hour <= 10:
                return 'morning'
            elif 11 <= hour <= 16:
                return 'midday'
            elif 17 <= hour <= 21:
                return 'evening'
            else:
                return 'night'
        
        self.combined_data['daytime'] = self.combined_data['hour'].apply(classify_daytime)
        
        # Filter out nighttime data for meaningful analysis
        daytime_data = self.combined_data[self.combined_data['daytime'] != 'night'].copy()
        
        # Hourly scaling factors
        hourly_factors = self.combined_data.groupby('hour')['scaling_factor'].agg([
            'count', 'mean', 'std', 'median', 'min', 'max'
        ]).round(4)
        
        # Daily scaling factors grouped by daytime
        daily_factors = daytime_data.groupby(['day_of_year', 'daytime'])['scaling_factor'].agg([
            'count', 'mean', 'median'
        ]).round(4)
        
        # Reset index to make day_of_year and daytime regular columns
        daily_factors = daily_factors.reset_index()
        
        # Monthly scaling factors
        monthly_factors = self.combined_data.groupby('month')['scaling_factor'].agg([
            'count', 'mean', 'std', 'median', 'min', 'max'
        ]).round(4)
        
        # Seasonal scaling factors
        seasonal_factors = self.combined_data.groupby('season')['scaling_factor'].agg([
            'count', 'mean', 'std', 'median', 'min', 'max'
        ]).round(4)
        
        # Overall statistics
        overall_stats = {
            'total_records': len(self.combined_data),
            'overall_mean_factor': self.combined_data['scaling_factor'].mean(),
            'overall_std_factor': self.combined_data['scaling_factor'].std(),
            'overall_median_factor': self.combined_data['scaling_factor'].median(),
            'correlation': self.combined_data['GS_10'].corr(self.combined_data['Irr']),
            'r_squared': self.combined_data['GS_10'].corr(self.combined_data['Irr']) ** 2
        }
        
        # Store results
        self.hourly_factors = hourly_factors
        self.daily_factors = daily_factors
        self.monthly_factors = monthly_factors
        self.seasonal_factors = seasonal_factors
        
        scaling_factors = {
            'hourly': hourly_factors,
            'daily': daily_factors,
            'monthly': monthly_factors,
            'seasonal': seasonal_factors,
            'overall': overall_stats
        }
        
        self.scaling_factors = scaling_factors
        return scaling_factors
    
    def create_visualizations(self):
        """
        Create comprehensive visualizations of the comparison analysis.
        """
        logger.info("Creating visualizations...")
        
        if self.combined_data is None:
            raise ValueError("Data must be merged first")
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Scatter plot: GS_10 vs Irr
        ax1 = plt.subplot(3, 3, 1)
        plt.scatter(self.combined_data['GS_10'], self.combined_data['Irr'], 
                   alpha=0.6, s=1)
        plt.plot([0, self.combined_data['GS_10'].max()], 
                [0, self.combined_data['GS_10'].max()], 'r--', alpha=0.8)
        plt.xlabel('DWD GS_10 (W/m²)')
        plt.ylabel('InfluxDB Irr (W/m²)')
        plt.title('GS_10 vs Irr Scatter Plot')
        plt.grid(True, alpha=0.3)
        
        # 2. Scaling factor distribution
        ax2 = plt.subplot(3, 3, 2)
        plt.hist(self.combined_data['scaling_factor'], bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Scaling Factor (Irr/GS_10)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Scaling Factors')
        plt.grid(True, alpha=0.3)
        
        # 3. Hourly scaling factors
        ax3 = plt.subplot(3, 3, 3)
        if self.hourly_factors is not None:
            plt.errorbar(self.hourly_factors.index, self.hourly_factors['mean'], 
                        yerr=self.hourly_factors['std'], fmt='o-', capsize=5)
            plt.xlabel('Hour of Day')
            plt.ylabel('Scaling Factor')
            plt.title('Scaling Factor by Hour (±1 std)')
            plt.grid(True, alpha=0.3)
            plt.xticks(range(0, 24, 2))
        
        # 4. Monthly scaling factors
        ax4 = plt.subplot(3, 3, 4)
        if self.monthly_factors is not None:
            plt.errorbar(self.monthly_factors.index, self.monthly_factors['mean'], 
                        yerr=self.monthly_factors['std'], fmt='o-', capsize=5)
            plt.xlabel('Month')
            plt.ylabel('Scaling Factor')
            plt.title('Scaling Factor by Month (±1 std)')
            plt.grid(True, alpha=0.3)
            plt.xticks(range(1, 13))
        
        # 5. Time series of scaling factors
        ax5 = plt.subplot(3, 3, 5)
        # Sample data for visualization (every 100th point to avoid overcrowding)
        sample_data = self.combined_data.iloc[::100].copy()
        plt.scatter(sample_data['timestamp'], sample_data['scaling_factor'], 
                   alpha=0.6, s=1)
        plt.xlabel('Time')
        plt.ylabel('Scaling Factor')
        plt.title('Scaling Factor Over Time (Sampled)')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # 6. Box plot by hour
        ax6 = plt.subplot(3, 3, 6)
        hour_data = [self.combined_data[self.combined_data['hour'] == h]['scaling_factor'].values 
                     for h in range(24)]
        plt.boxplot(hour_data, positions=range(24))
        plt.xlabel('Hour of Day')
        plt.ylabel('Scaling Factor')
        plt.title('Scaling Factor Distribution by Hour')
        plt.grid(True, alpha=0.3)
        plt.xticks(range(0, 24, 2))
        
        # 7. Seasonal comparison
        ax7 = plt.subplot(3, 3, 7)
        if self.seasonal_factors is not None:
            seasons = list(self.seasonal_factors.index)
            means = self.seasonal_factors['mean'].values
            stds = self.seasonal_factors['std'].values
            
            bars = plt.bar(seasons, means, yerr=stds, capsize=5, alpha=0.7)
            plt.xlabel('Season')
            plt.ylabel('Scaling Factor')
            plt.title('Scaling Factor by Season (±1 std)')
            plt.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, mean_val in zip(bars, means):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{mean_val:.3f}', ha='center', va='bottom')
        
        # 8. Correlation heatmap
        ax8 = plt.subplot(3, 3, 8)
        correlation_data = self.combined_data[['GS_10', 'Irr', 'scaling_factor']].corr()
        sns.heatmap(correlation_data, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.3f')
        plt.title('Correlation Matrix')
        
        # 9. Residual plot
        ax9 = plt.subplot(3, 3, 9)
        # Calculate residuals (difference between actual and predicted)
        predicted_irr = self.combined_data['GS_10'] * self.combined_data['scaling_factor'].mean()
        residuals = self.combined_data['Irr'] - predicted_irr
        
        plt.scatter(predicted_irr, residuals, alpha=0.6, s=1)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.8)
        plt.xlabel('Predicted Irr (W/m²)')
        plt.ylabel('Residuals (W/m²)')
        plt.title('Residual Plot')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'dwd_irr_comparison_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()  # Close figure to free memory
        
        logger.info(f"Visualizations saved to {self.output_dir}")
    
    def create_hourly_factor_table(self) -> pd.DataFrame:
        """
        Create a comprehensive hourly scaling factor table.
        
        Returns:
            DataFrame with hourly scaling factors and statistics
        """
        if self.hourly_factors is None:
            raise ValueError("Scaling factors must be calculated first")
        
        # Create enhanced hourly table
        hourly_table = self.hourly_factors.copy()
        
        # Add confidence intervals
        hourly_table['ci_95_lower'] = (hourly_table['mean'] - 1.96 * hourly_table['std']).round(4)
        hourly_table['ci_95_upper'] = (hourly_table['mean'] + 1.96 * hourly_table['std']).round(4)
        
        # Add coefficient of variation
        hourly_table['cv'] = (hourly_table['std'] / hourly_table['mean'] * 100).round(2)
        
        # Add recommended scaling factor (mean with outlier filtering)
        hourly_table['recommended_factor'] = hourly_table['median'].round(4)
        
        # Reorder columns for better readability
        column_order = [
            'count', 'mean', 'median', 'std', 'cv', 
            'ci_95_lower', 'ci_95_upper', 'min', 'max', 'recommended_factor'
        ]
        hourly_table = hourly_table[column_order]
        
        # Rename columns for better understanding
        hourly_table.columns = [
            'Records', 'Mean', 'Median', 'Std Dev', 'CV (%)', 
            '95% CI Lower', '95% CI Upper', 'Min', 'Max', 'Recommended'
        ]
        
        return hourly_table
    
    def create_daily_factor_table(self) -> pd.DataFrame:
        """
        Create a simplified daily scaling factor table with daytime grouping.
        
        Returns:
            DataFrame with daily scaling factors grouped by daytime
        """
        if self.daily_factors is None:
            raise ValueError("Daily scaling factors must be calculated first")
        
        # Create simplified daily table with only required columns
        daily_table = self.daily_factors.copy()
        
        # Ensure columns are in the correct order
        column_order = ['day_of_year', 'daytime', 'count', 'mean', 'median']
        daily_table = daily_table[column_order]
        
        # Rename columns for better understanding
        daily_table.columns = ['day_of_year', 'daytime', 'Records', 'Mean', 'Median']
        
        return daily_table
    
    def save_results(self):
        """
        Save all analysis results to files.
        """
        logger.info("Saving analysis results...")
        
        # Save scaling factors
        if self.scaling_factors:
            # Hourly factors
            hourly_table = self.create_hourly_factor_table()
            hourly_table.to_csv(self.output_dir / 'hourly_scaling_factors.csv')
            
            # Daily factors
            daily_table = self.create_daily_factor_table()
            daily_table.to_csv(self.output_dir / 'daily_scaling_factors.csv')
            
            # Monthly factors
            if self.monthly_factors is not None:
                self.monthly_factors.to_csv(self.output_dir / 'monthly_scaling_factors.csv')
            
            # Seasonal factors
            if self.seasonal_factors is not None:
                self.seasonal_factors.to_csv(self.output_dir / 'seasonal_scaling_factors.csv')
            
            # Overall statistics
            overall_df = pd.DataFrame([self.scaling_factors['overall']])
            overall_df.to_csv(self.output_dir / 'overall_statistics.csv', index=False)
        
        # Save combined data sample
        if self.combined_data is not None:
            # Save a sample for inspection
            sample_data = self.combined_data.sample(min(10000, len(self.combined_data)))
            sample_data.to_csv(self.output_dir / 'sample_combined_data.csv', index=False)
            
            # Save summary statistics
            summary_stats = self.combined_data.describe()
            summary_stats.to_csv(self.output_dir / 'data_summary_statistics.csv')
        
        logger.info(f"Results saved to {self.output_dir}")
    
    def create_html_report(self):
        """
        Create a comprehensive HTML report similar to other analyses.
        """
        logger.info("Creating HTML report...")
        
        if self.scaling_factors is None:
            logger.warning("No scaling factors available for HTML report")
            return
        
        # Prepare data for HTML report
        overall = self.scaling_factors['overall']
        
        # Create hourly factors table for HTML
        hourly_table_html = ""
        if self.hourly_factors is not None:
            hourly_table_html = self.hourly_factors.round(4).to_html(
                classes='table table-striped table-bordered',
                index=True,
                border=0
            )
        
        # Create daily factors table for HTML
        daily_table_html = ""
        if self.daily_factors is not None:
            daily_table_html = self.daily_factors.round(4).to_html(
                classes='table table-striped table-bordered',
                index=True,
                border=0
            )
        
        # Create monthly factors table for HTML
        monthly_table_html = ""
        if self.monthly_factors is not None:
            monthly_table_html = self.monthly_factors.round(4).to_html(
                classes='table table-striped table-bordered',
                index=True,
                border=0
            )
        
        # Create seasonal factors table for HTML
        seasonal_table_html = ""
        if self.seasonal_factors is not None:
            seasonal_table_html = self.seasonal_factors.round(4).to_html(
                classes='table table-striped table-bordered',
                index=True,
                border=0
            )
        
        # Create overall statistics table for HTML
        overall_table_html = pd.DataFrame([overall]).round(4).to_html(
            classes='table table-striped table-bordered',
            index=False,
            border=0
        )
        
        # Create sample data table for HTML
        sample_data_html = ""
        if self.combined_data is not None:
            sample_data = self.combined_data.sample(min(100, len(self.combined_data)))
            sample_data_html = sample_data.round(4).to_html(
                classes='table table-striped table-bordered',
                index=False,
                border=0
            )
        
        # HTML template
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DWD GS_10 vs InfluxDB Irr Comparison Analysis Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; text-align: center; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { color: #34495e; margin-top: 30px; border-left: 4px solid #3498db; padding-left: 15px; }
        h3 { color: #7f8c8d; margin-top: 25px; }
        .summary-box { background-color: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }
        .metric { display: inline-block; margin: 10px 20px; text-align: center; }
        .metric-value { font-size: 24px; font-weight: bold; color: #2980b9; }
        .metric-label { font-size: 14px; color: #7f8c8d; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #3498db; color: white; }
        tr:nth-child(even) { background-color: #f2f2f2; }
        .plot-container { text-align: center; margin: 20px 0; }
        .plot-container img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }
        .recommendations { background-color: #e8f5e8; padding: 15px; border-radius: 5px; margin: 20px 0; border-left: 4px solid #27ae60; }
        .warning { background-color: #fff3cd; padding: 15px; border-radius: 5px; margin: 20px 0; border-left: 4px solid #ffc107; }
        .info { background-color: #d1ecf1; padding: 15px; border-radius: 5px; margin: 20px 0; border-left: 4px solid #17a2b8; }
    </style>
</head>
<body>
    <div class="container">
        <h1>DWD GS_10 vs InfluxDB Irr Comparison Analysis Report</h1>
        
        <div class="summary-box">
            <h2>Analysis Summary</h2>
            <div class="metric">
                <div class="metric-value">{{ overall.total_records | int }}</div>
                <div class="metric-label">Total Records</div>
            </div>
            <div class="metric">
                <div class="metric-value">{{ "%.4f"|format(overall.overall_mean_factor) }}</div>
                <div class="metric-label">Mean Scaling Factor</div>
            </div>
            <div class="metric">
                <div class="metric-value">{{ "%.4f"|format(overall.correlation) }}</div>
                <div class="metric-label">Correlation</div>
            </div>
            <div class="metric">
                <div class="metric-value">{{ "%.4f"|format(overall.r_squared) }}</div>
                <div class="metric-label">R²</div>
            </div>
        </div>

        <div class="info">
            <h3>Analysis Information</h3>
            <p><strong>Analysis Date:</strong> {{ analysis_metadata.analysis_date }}</p>
            <p><strong>DWD Data Source:</strong> {{ analysis_metadata.dwd_csv_path }}</p>
            <p><strong>InfluxDB Data Source:</strong> {{ analysis_metadata.influx_csv_path }}</p>
            <p><strong>Output Directory:</strong> {{ analysis_metadata.output_directory }}</p>
        </div>

        <h2>Overall Statistics</h2>
        {{ overall_table_html | safe }}

        <h2>Hourly Scaling Factors</h2>
        <p>Scaling factors calculated for each hour of the day, showing temporal patterns in the relationship between DWD GS_10 and InfluxDB Irr values.</p>
        {{ hourly_table_html | safe }}

        <h2>Daily Scaling Factors</h2>
        <p>Scaling factors calculated for each day of the year, grouped by daytime periods (morning: 5-10h, midday: 11-16h, evening: 17-21h).</p>
        {{ daily_table_html | safe }}

        <h2>Monthly Scaling Factors</h2>
        <p>Scaling factors calculated for each month, showing seasonal variations in the relationship between the two data sources.</p>
        {{ monthly_table_html | safe }}

        <h2>Seasonal Scaling Factors</h2>
        <p>Scaling factors grouped by seasons, providing insights into long-term patterns.</p>
        {{ seasonal_table_html | safe }}

        <h2>Sample Combined Data</h2>
        <p>A sample of the merged DWD and InfluxDB data showing matched timestamps and calculated scaling factors.</p>
        {{ sample_data_html | safe }}

        <div class="plot-container">
            <h2>Analysis Visualizations</h2>
            <p>The comprehensive analysis plot has been saved as 'dwd_irr_comparison_analysis.png' in the output directory.</p>
            <p>This plot includes 9 different visualizations covering scatter plots, distributions, time series, and statistical analyses.</p>
        </div>

        <div class="recommendations">
            <h2>Key Findings & Recommendations</h2>
            <ul>
                <li><strong>Overall Scaling Factor:</strong> {{ "%.4f"|format(overall.overall_median_factor) }} (median)</li>
                <li><strong>Data Quality:</strong> {{ overall.total_records | int }} matched records provide a robust basis for analysis</li>
                <li><strong>Correlation Strength:</strong> {{ "%.4f"|format(overall.correlation) }} indicates the relationship strength between data sources</li>
                <li><strong>Variability:</strong> Standard deviation of {{ "%.4f"|format(overall.overall_std_factor) }} shows the spread of scaling factors</li>
            </ul>
        </div>

        <div class="warning">
            <h3>Usage Notes</h3>
            <ul>
                <li>Use hourly factors for time-dependent applications requiring high precision</li>
                <li>Use daily factors for day-specific applications and daily forecasting</li>
                <li>Consider seasonal variations for long-term forecasting models</li>
                <li>Validate scaling factors against your specific use case requirements</li>
                <li>Monitor data quality and update factors as new data becomes available</li>
            </ul>
        </div>

        <div class="info">
            <h3>Generated Files</h3>
            <ul>
                <li><strong>combined_dwd_irr_data.csv:</strong> Complete merged dataset</li>
                <li><strong>hourly_scaling_factors.csv:</strong> Hourly scaling factors with statistics</li>
                <li><strong>daily_scaling_factors.csv:</strong> Daily scaling factors with statistics</li>
                <li><strong>monthly_scaling_factors.csv:</strong> Monthly scaling factors</li>
                <li><strong>seasonal_scaling_factors.csv:</strong> Seasonal scaling factors</li>
                <li><strong>overall_statistics.csv:</strong> Overall analysis statistics</li>
                <li><strong>dwd_irr_comparison_analysis.png:</strong> Comprehensive visualization plot</li>
            </ul>
        </div>
    </div>
</body>
</html>
        """
        
        # Render HTML with data
        template = jinja2.Template(html_template)
        html_content = template.render(
            overall=overall,
            hourly_table_html=hourly_table_html,
            daily_table_html=daily_table_html,
            monthly_table_html=monthly_table_html,
            seasonal_table_html=seasonal_table_html,
            overall_table_html=overall_table_html,
            sample_data_html=sample_data_html,
            analysis_metadata=self.analysis_metadata
        )
        
        # Save HTML report
        html_path = self.output_dir / 'dwd_irr_comparison_report.html'
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML report saved to {html_path}")
    
    def print_summary_report(self):
        """
        Print a comprehensive summary report to console.
        """
        if self.scaling_factors is None:
            logger.warning("No scaling factors available for summary report")
            return
        
        print("\n" + "="*80)
        print("DWD GS_10 vs InfluxDB Irr COMPARISON ANALYSIS REPORT")
        print("="*80)
        
        # Overall statistics
        overall = self.scaling_factors['overall']
        print(f"\nOVERALL STATISTICS:")
        print(f"Total matched records: {overall['total_records']:,}")
        print(f"Overall scaling factor: {overall['overall_mean_factor']:.4f}")
        print(f"Standard deviation: {overall['overall_std_factor']:.4f}")
        print(f"Correlation coefficient: {overall['correlation']:.4f}")
        print(f"R-squared: {overall['r_squared']:.4f}")
        
        # Hourly summary
        if self.hourly_factors is not None:
            print(f"\nHOURLY SCALING FACTORS:")
            print(f"{'Hour':<4} {'Mean':<8} {'Median':<8} {'Std Dev':<8} {'Records':<8}")
            print("-" * 40)
            for hour in range(24):
                if hour in self.hourly_factors.index:
                    row = self.hourly_factors.loc[hour]
                    print(f"{hour:<4} {row['mean']:<8.4f} {row['median']:<8.4f} "
                          f"{row['std']:<8.4f} {row['count']:<8}")
        
        # Daily summary
        if self.daily_factors is not None:
            print(f"\nDAILY SCALING FACTORS (Grouped by Daytime):")
            print(f"{'Day':<4} {'Daytime':<8} {'Records':<8} {'Mean':<8} {'Median':<8}")
            print("-" * 45)
            # Show first 15 entries (5 days × 3 time periods)
            first_15_entries = self.daily_factors.head(15)
            
            for _, row in first_15_entries.iterrows():
                print(f"{row['day_of_year']:<4} {row['daytime']:<8} {row['count']:<8} "
                      f"{row['mean']:<8.4f} {row['median']:<8.4f}")
            
            if len(self.daily_factors) > 15:
                print("...")
                # Show last 15 entries
                last_15_entries = self.daily_factors.tail(15)
                for _, row in last_15_entries.iterrows():
                    print(f"{row['day_of_year']:<4} {row['daytime']:<8} {row['count']:<8} "
                          f"{row['mean']:<8.4f} {row['median']:<8.4f}")
        
        # Monthly summary
        if self.monthly_factors is not None:
            print(f"\nMONTHLY SCALING FACTORS:")
            print(f"{'Month':<6} {'Mean':<8} {'Median':<8} {'Std Dev':<8} {'Records':<8}")
            print("-" * 40)
            for month in range(1, 13):
                if month in self.monthly_factors.index:
                    row = self.monthly_factors.loc[month]
                    print(f"{month:<6} {row['mean']:<8.4f} {row['median']:<8.4f} "
                          f"{row['std']:<8.4f} {row['count']:<8}")
        
        # Seasonal summary
        if self.seasonal_factors is not None:
            print(f"\nSEASONAL SCALING FACTORS:")
            print(f"{'Season':<8} {'Mean':<8} {'Median':<8} {'Std Dev':<8} {'Records':<8}")
            print("-" * 40)
            for season in ['Winter', 'Spring', 'Summer', 'Autumn']:
                if season in self.seasonal_factors.index:
                    row = self.seasonal_factors.loc[season]
                    print(f"{season:<8} {row['mean']:<8.4f} {row['median']:<8.4f} "
                          f"{row['std']:<8.4f} {row['count']:<8}")
        
        print("\n" + "="*80)
        print("RECOMMENDATIONS:")
        print("="*80)
        
        # Provide recommendations
        if self.hourly_factors is not None:
            best_hour = self.hourly_factors['count'].idxmax()
            worst_hour = self.hourly_factors['std'].idxmax()
            
            print(f"• Best data coverage: Hour {best_hour} ({self.hourly_factors.loc[best_hour, 'count']} records)")
            print(f"• Highest variability: Hour {worst_hour} (std: {self.hourly_factors.loc[worst_hour, 'std']:.4f})")
        
        if self.daily_factors is not None:
            best_day = self.daily_factors['count'].idxmax()
            # Since we no longer have std in daily factors, we'll use count as a proxy for variability
            worst_day = self.daily_factors['count'].idxmin()
            print(f"• Best daily coverage: Day {best_day} ({self.daily_factors.loc[best_day, 'count']} records)")
            print(f"• Lowest daily coverage: Day {worst_day} ({self.daily_factors.loc[worst_day, 'count']} records)")
        
        if self.seasonal_factors is not None:
            best_season = self.seasonal_factors['count'].idxmax()
            print(f"• Best seasonal coverage: {best_season} ({self.seasonal_factors.loc[best_season, 'count']} records)")
        
        print(f"• Overall recommended scaling factor: {overall['overall_median_factor']:.4f}")
        print(f"• Use hourly factors for time-dependent applications")
        print(f"• Use daily factors for day-specific applications and daily forecasting")
        print(f"• Consider seasonal variations for long-term forecasting")
        
        print("\n" + "="*80)
        print(f"Results saved to: {self.output_dir}")
        print("="*80)
    
    def run_complete_analysis(self):
        """
        Run the complete analysis pipeline.
        """
        logger.info("Starting complete DWD vs InfluxDB Irr analysis...")
        
        try:
            # Auto-detect data files if not provided
            self.auto_detect_data_files()
            
            # Load data
            self.load_dwd_data()
            self.load_influx_data()
            
            # Merge and analyze
            self.merge_data()
            self.calculate_scaling_factors()
            
            # Create visualizations and save results
            self.create_visualizations()
            self.save_results()
            
            # Create HTML report
            self.create_html_report()
            
            # Print summary report
            self.print_summary_report()
            
            logger.info("Analysis completed successfully!")
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise


def main():
    """
    Main function to run the analysis from command line.
    """
    parser = argparse.ArgumentParser(
        description='Compare DWD GS_10 values with InfluxDB Irr values'
    )
    parser.add_argument(
        '--dwd-csv', 
        default=None,
        help='Path to DWD weather data CSV file (default: auto-detect from rawData directory)'
    )
    parser.add_argument(
        '--influx-csv', 
        default=None,
        help='Path to InfluxDB data CSV file (default: auto-detect from rawData directory)'
    )
    parser.add_argument(
        '--output-dir', 
        default='results/irr_comparison',
        help='Output directory for results (default: results/irr_comparison)'
    )
    
    args = parser.parse_args()
    
    # Create analyzer and run analysis
    analyzer = DWDIrrComparisonAnalyzer(
        dwd_csv_path=args.dwd_csv,
        influx_csv_path=args.influx_csv,
        output_dir=args.output_dir
    )
    
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()
