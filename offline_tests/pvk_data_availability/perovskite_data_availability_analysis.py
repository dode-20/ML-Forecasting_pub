#!/usr/bin/env python3
"""
Perovskite Data Availability Analysis - Precondition Assessment

This script performs a systematic data availability assessment for Silicon (Si) and 
Perovskite (Pvk) modules to ensure fair comparison windows for forecasting model evaluation.

Features:
- Heatmaps of daily data availability per module
- Aggregated counts of available modules per day for each technology  
- Identification of overlapping training, validation, and test windows
- Balanced seasonal coverage analysis
- Interactive HTML reports and summary files

"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import argparse
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set style for matplotlib
plt.style.use('default')
sns.set_palette("husl")

class PerovskiteDataAvailabilityAnalyzer:
    """
    Analyzer for comparing data availability between Silicon and Perovskite modules.
    """
    
    def __init__(self, silicon_path: str, perovskite_path: str, output_base_dir: str = None):
        """
        Initialize the analyzer with paths to clean data files.
        
        Args:
            silicon_path (str): Path to Silicon clean data CSV
            perovskite_path (str): Path to Perovskite clean data CSV
            output_base_dir (str): Base output directory (optional)
        """
        self.silicon_path = Path(silicon_path)
        self.perovskite_path = Path(perovskite_path)
        
        # Setup output directory with timestamp
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        timestamp_dir = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if output_base_dir:
            self.output_dir = Path(output_base_dir) / "perovskite_analysis" / "preconditions" / timestamp_dir
        else:
            self.output_dir = Path(__file__).parent.parent.parent / "results" / "perovskite_analysis" / "preconditions" / timestamp_dir
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.silicon_data = None
        self.perovskite_data = None
        self.analysis_results = {}
        
        print(f"Perovskite Data Availability Analysis initialized")
        print(f"Silicon data: {self.silicon_path}")
        print(f"Perovskite data: {self.perovskite_path}")
        print(f"Output directory: {self.output_dir}")
    
    def load_data(self) -> bool:
        """
        Load and validate the clean data files.
        
        Returns:
            bool: True if both files loaded successfully
        """
        print("\nStep 1: Loading clean data files...")
        
        try:
            # Load Silicon data
            if not self.silicon_path.exists():
                raise FileNotFoundError(f"Silicon data file not found: {self.silicon_path}")
            
            self.silicon_data = pd.read_csv(self.silicon_path)
            print(f"OK Silicon data loaded: {len(self.silicon_data)} records")
            
            # Load Perovskite data
            if not self.perovskite_path.exists():
                raise FileNotFoundError(f"Perovskite data file not found: {self.perovskite_path}")
            
            self.perovskite_data = pd.read_csv(self.perovskite_path)
            print(f"OK Perovskite data loaded: {len(self.perovskite_data)} records")
            
            # Validate and convert timestamps
            for name, data in [("Silicon", self.silicon_data), ("Perovskite", self.perovskite_data)]:
                if "_time" in data.columns:
                    data["_time"] = pd.to_datetime(data["_time"], errors='coerce')
                elif "timestamp" in data.columns:
                    data["timestamp"] = pd.to_datetime(data["timestamp"], errors='coerce')
                    data["_time"] = data["timestamp"]
                else:
                    raise ValueError(f"No timestamp column found in {name} data")
                
                print(f"OK {name} timestamps converted")
            
            return True
            
        except Exception as e:
            print(f"FAIL Failed to load data: {e}")
            return False
    
    def analyze_data_availability(self):
        """
        Perform comprehensive data availability analysis.
        """
        print("\nStep 2: Analyzing data availability...")
        
        # Basic statistics
        self._analyze_basic_statistics()
        
        # Daily availability analysis
        self._analyze_daily_availability()
        
        # Module availability analysis
        self._analyze_module_availability()
        
        # Overlap analysis
        self._analyze_temporal_overlap()
        
        # Seasonal coverage analysis
        self._analyze_seasonal_coverage()
        
        print("OK Data availability analysis completed")
    
    def _analyze_basic_statistics(self):
        """Analyze basic dataset statistics."""
        print("  - Computing basic statistics...")
        
        silicon_stats = {
            "total_records": len(self.silicon_data),
            "date_range": (
                self.silicon_data["_time"].min(),
                self.silicon_data["_time"].max()
            ),
            "unique_modules": self.silicon_data["Name"].nunique() if "Name" in self.silicon_data.columns else 1,
            "data_resolution": self._detect_resolution(self.silicon_data),
            "missing_values": self.silicon_data.isnull().sum().to_dict()
        }
        
        perovskite_stats = {
            "total_records": len(self.perovskite_data),
            "date_range": (
                self.perovskite_data["_time"].min(),
                self.perovskite_data["_time"].max()
            ),
            "unique_modules": self.perovskite_data["Name"].nunique() if "Name" in self.perovskite_data.columns else 1,
            "data_resolution": self._detect_resolution(self.perovskite_data),
            "missing_values": self.perovskite_data.isnull().sum().to_dict()
        }
        
        self.analysis_results["basic_stats"] = {
            "silicon": silicon_stats,
            "perovskite": perovskite_stats
        }
    
    def _detect_resolution(self, data: pd.DataFrame) -> str:
        """Detect the temporal resolution of the data."""
        if len(data) < 2:
            return "unknown"
        
        # Calculate time differences
        time_diffs = data["_time"].diff().dropna()
        most_common_diff = time_diffs.mode().iloc[0] if not time_diffs.empty else None
        
        if most_common_diff:
            if most_common_diff <= pd.Timedelta(minutes=5):
                return "5min"
            elif most_common_diff <= pd.Timedelta(minutes=10):
                return "10min"
            elif most_common_diff <= pd.Timedelta(hours=1):
                return "1h"
            else:
                return f"{most_common_diff}"
        
        return "unknown"
    
    def _analyze_daily_availability(self):
        """Analyze data availability on a daily basis."""
        print("  - Analyzing daily availability...")
        
        # Create daily availability matrices
        silicon_daily = self._create_daily_availability_matrix(self.silicon_data, "Silicon")
        perovskite_daily = self._create_daily_availability_matrix(self.perovskite_data, "Perovskite")
        
        self.analysis_results["daily_availability"] = {
            "silicon": silicon_daily,
            "perovskite": perovskite_daily
        }
    
    def _create_daily_availability_matrix(self, data: pd.DataFrame, tech_name: str) -> pd.DataFrame:
        """Create a daily availability matrix for heatmap visualization."""
        # Extract date and module information
        data_copy = data.copy()
        data_copy["date"] = data_copy["_time"].dt.date
        
        if "Name" in data_copy.columns:
            # Multi-module case
            daily_counts = data_copy.groupby(["date", "Name"]).size().reset_index(name="record_count")
            availability_matrix = daily_counts.pivot(index="date", columns="Name", values="record_count")
        else:
            # Single module case (averaged data)
            daily_counts = data_copy.groupby("date").size().reset_index(name="record_count")
            availability_matrix = daily_counts.set_index("date")
            availability_matrix.columns = [f"{tech_name}_avg"]
        
        # Fill NaN with 0 and convert to binary availability (0/1)
        availability_matrix = availability_matrix.fillna(0)
        availability_binary = (availability_matrix > 0).astype(int)
        
        return availability_binary
    
    def _analyze_module_availability(self):
        """Analyze module-level availability statistics."""
        print("  - Analyzing module availability...")
        
        silicon_modules = self._get_module_statistics(self.silicon_data, "Silicon")
        perovskite_modules = self._get_module_statistics(self.perovskite_data, "Perovskite")
        
        self.analysis_results["module_availability"] = {
            "silicon": silicon_modules,
            "perovskite": perovskite_modules
        }
    
    def _get_module_statistics(self, data: pd.DataFrame, tech_name: str) -> Dict:
        """Get detailed statistics for individual modules."""
        if "Name" not in data.columns:
            # Averaged data case
            return {
                "module_count": 1,
                "module_names": [f"{tech_name}_average"],
                "total_days": len(data["_time"].dt.date.unique()),
                "avg_records_per_day": len(data) / len(data["_time"].dt.date.unique()) if len(data) > 0 else 0
            }
        
        modules = data["Name"].unique()
        module_stats = []
        
        for module in modules:
            module_data = data[data["Name"] == module]
            stats = {
                "name": module,
                "total_records": len(module_data),
                "date_range": (
                    module_data["_time"].min(),
                    module_data["_time"].max()
                ),
                "active_days": len(module_data["_time"].dt.date.unique()),
                "avg_records_per_day": len(module_data) / len(module_data["_time"].dt.date.unique()) if len(module_data) > 0 else 0
            }
            module_stats.append(stats)
        
        return {
            "module_count": len(modules),
            "module_names": modules.tolist(),
            "module_details": module_stats,
            "total_active_days": len(data["_time"].dt.date.unique())
        }
    
    def _analyze_temporal_overlap(self):
        """Analyze temporal overlap between Silicon and Perovskite data."""
        print("  - Analyzing temporal overlap...")
        
        # Get date ranges
        si_dates = set(self.silicon_data["_time"].dt.date)
        pvk_dates = set(self.perovskite_data["_time"].dt.date)
        
        # Calculate overlaps
        overlap_dates = si_dates.intersection(pvk_dates)
        si_only_dates = si_dates - pvk_dates
        pvk_only_dates = pvk_dates - si_dates
        
        # Identify continuous periods
        overlap_periods = self._identify_continuous_periods(sorted(overlap_dates))
        
        self.analysis_results["temporal_overlap"] = {
            "total_overlap_days": len(overlap_dates),
            "silicon_only_days": len(si_only_dates),
            "perovskite_only_days": len(pvk_only_dates),
            "overlap_percentage": len(overlap_dates) / len(si_dates.union(pvk_dates)) * 100,
            "overlap_periods": overlap_periods,
            "overlapping_date_range": (
                min(overlap_dates) if overlap_dates else None,
                max(overlap_dates) if overlap_dates else None
            )
        }
    
    def _identify_continuous_periods(self, dates: List) -> List[Dict]:
        """Identify continuous periods from a list of dates."""
        if not dates:
            return []
        
        periods = []
        start_date = dates[0]
        end_date = dates[0]
        
        for i in range(1, len(dates)):
            current_date = dates[i]
            if (current_date - end_date).days == 1:
                # Continuous period continues
                end_date = current_date
            else:
                # Gap found, save current period and start new one
                periods.append({
                    "start": start_date,
                    "end": end_date,
                    "duration_days": (end_date - start_date).days + 1
                })
                start_date = current_date
                end_date = current_date
        
        # Add the last period
        periods.append({
            "start": start_date,
            "end": end_date,
            "duration_days": (end_date - start_date).days + 1
        })
        
        return periods
    
    def _analyze_seasonal_coverage(self):
        """Analyze seasonal coverage for both technologies."""
        print("  - Analyzing seasonal coverage...")
        
        # Define seasons
        def get_season(date):
            month = date.month
            if month in [12, 1, 2]:
                return "Winter"
            elif month in [3, 4, 5]:
                return "Spring"
            elif month in [6, 7, 8]:
                return "Summer"
            else:
                return "Autumn"
        
        # Analyze seasonal coverage
        silicon_seasonal = self._get_seasonal_coverage(self.silicon_data, get_season)
        perovskite_seasonal = self._get_seasonal_coverage(self.perovskite_data, get_season)
        
        self.analysis_results["seasonal_coverage"] = {
            "silicon": silicon_seasonal,
            "perovskite": perovskite_seasonal
        }
    
    def _get_seasonal_coverage(self, data: pd.DataFrame, season_func) -> Dict:
        """Get seasonal coverage statistics."""
        data_copy = data.copy()
        data_copy["season"] = data_copy["_time"].dt.date.apply(season_func)
        data_copy["date"] = data_copy["_time"].dt.date
        
        seasonal_stats = {}
        for season in ["Winter", "Spring", "Summer", "Autumn"]:
            season_data = data_copy[data_copy["season"] == season]
            seasonal_stats[season] = {
                "total_records": len(season_data),
                "unique_dates": len(season_data["date"].unique()),
                "date_range": (
                    season_data["_time"].min() if len(season_data) > 0 else None,
                    season_data["_time"].max() if len(season_data) > 0 else None
                )
            }
        
        return seasonal_stats
    
    def create_heatmaps(self):
        """Create heatmaps for data availability visualization."""
        print("\nStep 3: Creating heatmaps...")
        
        # Static heatmaps with matplotlib/seaborn
        self._create_static_heatmaps()
        
        # Interactive heatmaps with plotly
        self._create_interactive_heatmaps()
        
        print("OK Heatmaps created")
    
    def _create_static_heatmaps(self):
        """Create static calendar-style heatmaps using matplotlib and seaborn."""
        print("  - Creating calendar-style heatmaps...")
        
        # Create calendar heatmaps for both technologies
        self._create_calendar_heatmap("silicon", self.silicon_data)
        self._create_calendar_heatmap("perovskite", self.perovskite_data)
        
        # Create combined comparison plot
        self._create_combined_calendar_comparison()
        
        print(f"    Calendar-style heatmaps saved")
    
    def _create_calendar_heatmap(self, tech_name: str, data: pd.DataFrame):
        """
        Create a calendar-style heatmap showing daily timestamp counts.
        
        Args:
            tech_name (str): Technology name ('silicon' or 'perovskite')
            data (pd.DataFrame): Data for the technology
        """
        if data.empty:
            print(f"    No data available for {tech_name}")
            return
        
        # Calculate daily timestamp counts
        data_copy = data.copy()
        data_copy["date"] = data_copy["_time"].dt.date
        daily_counts = data_copy.groupby("date").size()
        
        # Get full date range
        min_date = min(daily_counts.index)
        max_date = max(daily_counts.index)
        
        # Create calendar layout
        calendar_data = self._create_calendar_matrix(daily_counts, min_date, max_date)
        
        # Create the heatmap
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Create heatmap with custom colormap
        im = ax.imshow(calendar_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=24)
        
        # Set title
        tech_title = tech_name.capitalize()
        ax.set_title(f'{tech_title} Modules - Daily Timestamp Availability\n'
                    f'(Calendar View: {min_date} to {max_date})', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Configure axes
        weeks, dates_matrix = self._get_calendar_labels(min_date, max_date)
        
        # Set x-axis (days of week)
        ax.set_xticks(range(7))
        ax.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        ax.set_xlabel('Day of Week', fontsize=12)
        
        # Set y-axis (weeks)
        ax.set_yticks(range(0, len(weeks), max(1, len(weeks)//10)))  # Show max 10 labels
        ax.set_yticklabels([f'Week {i+1}' for i in range(0, len(weeks), max(1, len(weeks)//10))])
        ax.set_ylabel('Week Number', fontsize=12)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Timestamps per Day (max: 24 for 1h resolution)', rotation=270, labelpad=20)
        
        # Add grid
        ax.set_xticks(np.arange(-0.5, 7, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(weeks), 1), minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=2)
        
        # Add text annotations for values
        self._add_calendar_annotations(ax, calendar_data, dates_matrix)
        
        plt.tight_layout()
        
        # Save heatmap
        heatmap_path = self.output_dir / f"calendar_heatmap_{tech_name}.png"
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    {tech_title} calendar heatmap saved: {heatmap_path}")
    
    def _create_calendar_matrix(self, daily_counts: pd.Series, min_date, max_date):
        """Create a matrix representing the calendar layout."""
        # Generate all dates in range
        current_date = min_date
        all_dates = []
        while current_date <= max_date:
            all_dates.append(current_date)
            current_date += timedelta(days=1)
        
        # Find the first Monday before or on min_date
        start_date = min_date
        while start_date.weekday() != 0:  # 0 = Monday
            start_date -= timedelta(days=1)
        
        # Calculate matrix dimensions
        total_days = (max_date - start_date).days + 1
        weeks_needed = (total_days + 6) // 7  # Round up
        
        # Create matrix (weeks x days_of_week)
        calendar_matrix = np.full((weeks_needed, 7), np.nan)
        
        # Fill matrix with timestamp counts
        current = start_date
        for week in range(weeks_needed):
            for day in range(7):
                if current in daily_counts.index:
                    calendar_matrix[week, day] = daily_counts[current]
                elif min_date <= current <= max_date:
                    calendar_matrix[week, day] = 0  # No data for this day
                current += timedelta(days=1)
        
        return calendar_matrix
    
    def _get_calendar_labels(self, min_date, max_date):
        """Get labels for calendar axes."""
        # Find the first Monday
        start_date = min_date
        while start_date.weekday() != 0:
            start_date -= timedelta(days=1)
        
        # Calculate weeks
        total_days = (max_date - start_date).days + 1
        weeks_needed = (total_days + 6) // 7
        weeks = list(range(weeks_needed))
        
        # Create dates matrix for annotations
        dates_matrix = []
        current = start_date
        for week in range(weeks_needed):
            week_dates = []
            for day in range(7):
                if min_date <= current <= max_date:
                    week_dates.append(current)
                else:
                    week_dates.append(None)
                current += timedelta(days=1)
            dates_matrix.append(week_dates)
        
        return weeks, dates_matrix
    
    def _add_calendar_annotations(self, ax, calendar_data, dates_matrix):
        """Add date and value annotations to calendar heatmap."""
        for week in range(calendar_data.shape[0]):
            for day in range(calendar_data.shape[1]):
                value = calendar_data[week, day]
                date_obj = dates_matrix[week][day] if week < len(dates_matrix) and day < len(dates_matrix[week]) else None
                
                if date_obj and not np.isnan(value):
                    # Format date as DD.MM
                    date_str = f"{date_obj.day:02d}.{date_obj.month:02d}"
                    
                    # Combine date and timestamp count in one line
                    combined_text = f"{date_str} - {int(value)}"
                    
                    # Choose text color based on background
                    color = 'white' if value < 12 else 'black'  # Contrast color
                    
                    ax.text(day, week, combined_text, 
                           ha='center', va='center', fontsize=6, fontweight='bold', color=color)
    
    def _create_combined_calendar_comparison(self):
        """Create a comprehensive comparison including overlap analysis."""
        # Calculate daily counts for both technologies
        si_daily = self._get_daily_timestamp_counts(self.silicon_data, "Silicon")
        pvk_daily = self._get_daily_timestamp_counts(self.perovskite_data, "Perovskite")
        
        if si_daily.empty and pvk_daily.empty:
            print("    No data available for comparison")
            return
        
        # Get combined date range
        all_dates = set()
        if not si_daily.empty:
            all_dates.update(si_daily.index)
        if not pvk_daily.empty:
            all_dates.update(pvk_daily.index)
        
        min_date = min(all_dates)
        max_date = max(all_dates)
        
        # Create combined figure with 3 subplots
        fig = plt.figure(figsize=(26, 16))
        
        # Create grid layout: 2 rows, with colorbars on the right
        gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.05], height_ratios=[1, 1], 
                             hspace=0.3, wspace=0.15)
        
        # Top row: Side-by-side heatmaps
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Silicon subplot
        if not si_daily.empty:
            si_matrix = self._create_calendar_matrix(si_daily, min_date, max_date)
            im1 = ax1.imshow(si_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=24)
            ax1.set_title('Silicon Modules', fontsize=14, fontweight='bold')
            self._setup_calendar_axes(ax1, min_date, max_date)
        
        # Perovskite subplot
        if not pvk_daily.empty:
            pvk_matrix = self._create_calendar_matrix(pvk_daily, min_date, max_date)
            im2 = ax2.imshow(pvk_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=24)
            ax2.set_title('Perovskite Modules', fontsize=14, fontweight='bold')
            self._setup_calendar_axes(ax2, min_date, max_date)
        
        # Add colorbar for heatmaps (right side)
        if not si_daily.empty or not pvk_daily.empty:
            cbar_ax1 = fig.add_subplot(gs[0, 2])
            cbar1 = fig.colorbar(im1 if not si_daily.empty else im2, 
                               cax=cbar_ax1, orientation='vertical')
            cbar1.set_label('Timestamps per Day\n(max: 24 for 1h resolution)', 
                          rotation=270, labelpad=25)
        
        # Bottom row: Overlap analysis (spanning all columns except colorbar)
        ax3 = fig.add_subplot(gs[1, :2])
        overlap_im = self._create_overlap_calendar_plot(ax3, si_daily, pvk_daily, min_date, max_date)
        
        # Add colorbar for overlap plot
        if overlap_im is not None:
            cbar_ax2 = fig.add_subplot(gs[1, 2])
            cbar2 = fig.colorbar(overlap_im, cax=cbar_ax2, orientation='vertical', 
                               ticks=[0, 1])
            cbar2.set_ticklabels(['No Overlap', 'Both ≥5 timestamps'])
            cbar2.set_label('Data Overlap Status', rotation=270, labelpad=25)
        
        plt.tight_layout()
        
        # Save comparison
        comparison_path = self.output_dir / "calendar_comparison_heatmap.png"
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    Combined comparison with overlap analysis saved: {comparison_path}")
    
    def _create_overlap_calendar_plot(self, ax, si_daily, pvk_daily, min_date, max_date):
        """Create a calendar plot showing days with min 5 timestamps for both technologies."""
        # Create overlap matrix
        overlap_matrix = self._create_overlap_matrix(si_daily, pvk_daily, min_date, max_date, min_threshold=5)
        weeks, dates_matrix = self._get_calendar_labels(min_date, max_date)
        
        # Create binary colormap (white = no overlap, blue = overlap)
        from matplotlib.colors import ListedColormap
        colors = ['white', '#2E86AB']  # White for no overlap, blue for overlap
        cmap = ListedColormap(colors)
        
        # Create the plot
        im = ax.imshow(overlap_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)
        
        # Set title
        ax.set_title('Data Overlap Analysis (min 5 timestamps for both technologies)', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Setup axes
        self._setup_calendar_axes(ax, min_date, max_date)
        
        # Add grid
        ax.set_xticks(np.arange(-0.5, 7, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(weeks), 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=1, alpha=0.5)
        
        # Add annotations for overlap days
        self._add_overlap_annotations(ax, overlap_matrix, dates_matrix, si_daily, pvk_daily)
        
        # Return the image object for external colorbar creation
        return im
    
    def _create_overlap_matrix(self, si_daily, pvk_daily, min_date, max_date, min_threshold=5):
        """Create a matrix showing days where both technologies have min_threshold timestamps."""
        # Find the first Monday
        start_date = min_date
        while start_date.weekday() != 0:
            start_date -= timedelta(days=1)
        
        # Calculate matrix dimensions
        total_days = (max_date - start_date).days + 1
        weeks_needed = (total_days + 6) // 7
        
        # Create matrix
        overlap_matrix = np.full((weeks_needed, 7), np.nan)
        
        # Fill matrix with overlap information
        current = start_date
        for week in range(weeks_needed):
            for day in range(7):
                if min_date <= current <= max_date:
                    si_count = si_daily.get(current, 0) if not si_daily.empty else 0
                    pvk_count = pvk_daily.get(current, 0) if not pvk_daily.empty else 0
                    
                    # Check if both technologies have at least min_threshold timestamps
                    if si_count >= min_threshold and pvk_count >= min_threshold:
                        overlap_matrix[week, day] = 1  # Overlap
                    else:
                        overlap_matrix[week, day] = 0  # No overlap
                current += timedelta(days=1)
        
        return overlap_matrix
    
    def _add_overlap_annotations(self, ax, overlap_matrix, dates_matrix, si_daily, pvk_daily):
        """Add annotations to overlap calendar plot."""
        for week in range(overlap_matrix.shape[0]):
            for day in range(overlap_matrix.shape[1]):
                value = overlap_matrix[week, day]
                date_obj = dates_matrix[week][day] if week < len(dates_matrix) and day < len(dates_matrix[week]) else None
                
                if date_obj and not np.isnan(value):
                    # Format date as DD.MM
                    date_str = f"{date_obj.day:02d}.{date_obj.month:02d}"
                    
                    if value == 1:  # Overlap day
                        # Get timestamp counts
                        si_count = si_daily.get(date_obj, 0) if not si_daily.empty else 0
                        pvk_count = pvk_daily.get(date_obj, 0) if not pvk_daily.empty else 0
                        
                        # Combine all info in one line: date - Si:count - Pvk:count
                        combined_text = f"{date_str} - Si:{si_count} - Pvk:{pvk_count}"
                        
                        ax.text(day, week, combined_text, 
                               ha='center', va='center', fontsize=5, color='white', fontweight='bold')
                    else:  # No overlap day
                        # Just show the date
                        ax.text(day, week, date_str, 
                               ha='center', va='center', fontsize=6, fontweight='bold', color='black')
    
    def _get_daily_timestamp_counts(self, data: pd.DataFrame, tech_name: str) -> pd.Series:
        """Get daily timestamp counts for a technology."""
        if data.empty:
            return pd.Series(dtype=int)
        
        data_copy = data.copy()
        data_copy["date"] = data_copy["_time"].dt.date
        return data_copy.groupby("date").size()
    
    def _setup_calendar_axes(self, ax, min_date, max_date):
        """Setup axes for calendar heatmap."""
        weeks, _ = self._get_calendar_labels(min_date, max_date)
        
        # Set x-axis (days of week)
        ax.set_xticks(range(7))
        ax.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        ax.set_xlabel('Day of Week', fontsize=12)
        
        # Set y-axis (weeks)
        ax.set_yticks(range(0, len(weeks), max(1, len(weeks)//8)))
        ax.set_yticklabels([f'W{i+1}' for i in range(0, len(weeks), max(1, len(weeks)//8))])
        ax.set_ylabel('Week Number', fontsize=12)
        
        # Add grid
        ax.set_xticks(np.arange(-0.5, 7, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(weeks), 1), minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=1)
    
    def _plot_aggregated_counts(self, ax1, ax2):
        """Plot aggregated daily counts."""
        # Silicon daily aggregation
        si_data = self.analysis_results["daily_availability"]["silicon"]
        if not si_data.empty:
            daily_counts_si = si_data.sum(axis=1)
            ax1.plot(si_data.index, daily_counts_si, color='blue', linewidth=2, label='Silicon')
            ax1.set_title("Silicon - Active Modules per Day", fontweight='bold')
            ax1.set_xlabel("Date")
            ax1.set_ylabel("Number of Active Modules")
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='x', rotation=45)
        
        # Perovskite daily aggregation
        pvk_data = self.analysis_results["daily_availability"]["perovskite"]
        if not pvk_data.empty:
            daily_counts_pvk = pvk_data.sum(axis=1)
            ax2.plot(pvk_data.index, daily_counts_pvk, color='red', linewidth=2, label='Perovskite')
            ax2.set_title("Perovskite - Active Modules per Day", fontweight='bold')
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Number of Active Modules")
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis='x', rotation=45)
    
    def _create_interactive_heatmaps(self):
        """Create interactive calendar-style heatmaps using plotly."""
        print("  - Creating interactive calendar heatmaps...")
        
        # Create interactive calendar heatmaps
        self._create_interactive_calendar_heatmap("silicon", self.silicon_data)
        self._create_interactive_calendar_heatmap("perovskite", self.perovskite_data)
        
        # Create combined interactive comparison
        self._create_interactive_calendar_comparison()
        
        print(f"    Interactive calendar heatmaps saved")
    
    def _create_interactive_calendar_heatmap(self, tech_name: str, data: pd.DataFrame):
        """Create an interactive calendar heatmap for a single technology."""
        if data.empty:
            return
        
        # Calculate daily timestamp counts
        daily_counts = self._get_daily_timestamp_counts(data, tech_name)
        if daily_counts.empty:
            return
        
        # Get date range
        min_date = min(daily_counts.index)
        max_date = max(daily_counts.index)
        
        # Create calendar matrix
        calendar_matrix = self._create_calendar_matrix(daily_counts, min_date, max_date)
        weeks, dates_matrix = self._get_calendar_labels(min_date, max_date)
        
        # Prepare data for plotly
        z_data = calendar_matrix
        x_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        y_labels = [f'Week {i+1}' for i in range(len(weeks))]
        
        # Create hover text with dates and values
        hover_text = []
        for week in range(calendar_matrix.shape[0]):
            hover_week = []
            for day in range(calendar_matrix.shape[1]):
                value = calendar_matrix[week, day]
                date_obj = dates_matrix[week][day] if week < len(dates_matrix) and day < len(dates_matrix[week]) else None
                
                if date_obj and not np.isnan(value):
                    hover_week.append(f"Date: {date_obj}<br>Timestamps: {int(value)}/24<br>Coverage: {value/24*100:.1f}%")
                else:
                    hover_week.append("No data")
            hover_text.append(hover_week)
        
        # Create figure
        fig = go.Figure(data=go.Heatmap(
            z=z_data,
            x=x_labels,
            y=y_labels,
            colorscale='RdYlGn',
            zmin=0,
            zmax=24,
            hoverinfo='text',
            text=hover_text,
            colorbar=dict(
                title="Timestamps per Day"
            )
        ))
        
        # Update layout
        tech_title = tech_name.capitalize()
        fig.update_layout(
            title=f'{tech_title} Modules - Daily Timestamp Availability<br>'
                  f'<sub>Calendar View: {min_date} to {max_date}</sub>',
            xaxis_title="Day of Week",
            yaxis_title="Week Number",
            width=1000,
            height=600,
            yaxis=dict(autorange='reversed')  # Start from top
        )
        
        # Save interactive heatmap
        interactive_path = self.output_dir / f"interactive_calendar_{tech_name}.html"
        pyo.plot(fig, filename=str(interactive_path), auto_open=False)
        
        print(f"    {tech_title} interactive calendar saved: {interactive_path}")
    
    def _create_interactive_calendar_comparison(self):
        """Create an interactive side-by-side calendar comparison."""
        # Calculate daily counts
        si_daily = self._get_daily_timestamp_counts(self.silicon_data, "Silicon")
        pvk_daily = self._get_daily_timestamp_counts(self.perovskite_data, "Perovskite")
        
        if si_daily.empty and pvk_daily.empty:
            return
        
        # Get combined date range
        all_dates = set()
        if not si_daily.empty:
            all_dates.update(si_daily.index)
        if not pvk_daily.empty:
            all_dates.update(pvk_daily.index)
        
        min_date = min(all_dates)
        max_date = max(all_dates)
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Silicon Modules', 'Perovskite Modules'),
            horizontal_spacing=0.1
        )
        
        # Silicon heatmap
        if not si_daily.empty:
            si_matrix = self._create_calendar_matrix(si_daily, min_date, max_date)
            weeks, _ = self._get_calendar_labels(min_date, max_date)
            
            fig.add_trace(
                go.Heatmap(
                    z=si_matrix,
                    x=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                    y=[f'W{i+1}' for i in range(len(weeks))],
                    colorscale='RdYlGn',
                    zmin=0,
                    zmax=24,
                    name="Silicon",
                    hovertemplate="Day: %{x}<br>Week: %{y}<br>Timestamps: %{z}/24<extra></extra>",
                    colorbar=dict(x=0.45, title="Timestamps/Day")
                ),
                row=1, col=1
            )
        
        # Perovskite heatmap
        if not pvk_daily.empty:
            pvk_matrix = self._create_calendar_matrix(pvk_daily, min_date, max_date)
            weeks, _ = self._get_calendar_labels(min_date, max_date)
            
            fig.add_trace(
                go.Heatmap(
                    z=pvk_matrix,
                    x=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                    y=[f'W{i+1}' for i in range(len(weeks))],
                    colorscale='RdYlGn',
                    zmin=0,
                    zmax=24,
                    name="Perovskite",
                    hovertemplate="Day: %{x}<br>Week: %{y}<br>Timestamps: %{z}/24<extra></extra>",
                    colorbar=dict(x=1.05, title="Timestamps/Day")
                ),
                row=1, col=2
            )
        
        # Update layout
        fig.update_layout(
            title=f'Technology Comparison - Daily Timestamp Availability<br>'
                  f'<sub>Calendar View: {min_date} to {max_date}</sub>',
            width=1400,
            height=600,
            showlegend=False
        )
        
        # Update y-axes to start from top
        fig.update_yaxes(autorange='reversed')
        
        # Save interactive comparison
        comparison_path = self.output_dir / "interactive_calendar_comparison.html"
        pyo.plot(fig, filename=str(comparison_path), auto_open=False)
        
        print(f"    Interactive comparison saved: {comparison_path}")
    
    def create_overlap_analysis_plots(self):
        """Create plots specifically for temporal overlap analysis."""
        print("\nStep 4: Creating overlap analysis plots...")
        
        # Temporal overlap visualization
        overlap_data = self.analysis_results["temporal_overlap"]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle("Temporal Overlap Analysis: Silicon vs Perovskite", fontsize=14, fontweight='bold')
        
        # Coverage comparison pie chart
        labels = ['Overlap', 'Silicon Only', 'Perovskite Only']
        sizes = [
            overlap_data["total_overlap_days"],
            overlap_data["silicon_only_days"],
            overlap_data["perovskite_only_days"]
        ]
        colors = ['green', 'blue', 'red']
        
        axes[0, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title("Data Coverage Distribution")
        
        # Seasonal coverage comparison
        seasonal_data = self.analysis_results["seasonal_coverage"]
        seasons = ["Winter", "Spring", "Summer", "Autumn"]
        
        si_seasonal = [seasonal_data["silicon"][s]["unique_dates"] for s in seasons]
        pvk_seasonal = [seasonal_data["perovskite"][s]["unique_dates"] for s in seasons]
        
        x = np.arange(len(seasons))
        width = 0.35
        
        axes[0, 1].bar(x - width/2, si_seasonal, width, label='Silicon', color='blue', alpha=0.7)
        axes[0, 1].bar(x + width/2, pvk_seasonal, width, label='Perovskite', color='red', alpha=0.7)
        axes[0, 1].set_xlabel('Season')
        axes[0, 1].set_ylabel('Days with Data')
        axes[0, 1].set_title('Seasonal Data Coverage')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(seasons)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Overlap periods timeline
        if overlap_data["overlap_periods"]:
            periods = overlap_data["overlap_periods"]
            starts = [p["start"] for p in periods]
            durations = [p["duration_days"] for p in periods]
            
            axes[1, 0].barh(range(len(periods)), durations, color='green', alpha=0.7)
            axes[1, 0].set_yticks(range(len(periods)))
            axes[1, 0].set_yticklabels([f"Period {i+1}" for i in range(len(periods))])
            axes[1, 0].set_xlabel('Duration (Days)')
            axes[1, 0].set_title('Continuous Overlap Periods')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Data quality comparison
        si_stats = self.analysis_results["basic_stats"]["silicon"]
        pvk_stats = self.analysis_results["basic_stats"]["perovskite"]
        
        metrics = ['Total Records', 'Unique Modules', 'Time Span (Days)']
        si_values = [
            si_stats["total_records"],
            si_stats["unique_modules"],
            (si_stats["date_range"][1] - si_stats["date_range"][0]).days
        ]
        pvk_values = [
            pvk_stats["total_records"],
            pvk_stats["unique_modules"],
            (pvk_stats["date_range"][1] - pvk_stats["date_range"][0]).days
        ]
        
        x = np.arange(len(metrics))
        axes[1, 1].bar(x - width/2, si_values, width, label='Silicon', color='blue', alpha=0.7)
        axes[1, 1].bar(x + width/2, pvk_values, width, label='Perovskite', color='red', alpha=0.7)
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Dataset Comparison')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(metrics, rotation=45)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save overlap analysis
        overlap_path = self.output_dir / "temporal_overlap_analysis.png"
        plt.savefig(overlap_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"OK Overlap analysis plots saved: {overlap_path}")
    
    def generate_html_report(self):
        """Generate a comprehensive HTML report with all plots and analysis."""
        print("\nStep 5: Generating HTML report...")
        
        # HTML template
        html_content = self._create_html_template()
        
        # Save HTML report
        html_path = self.output_dir / "precondition_analysis_report.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"    HTML report saved: {html_path}")
    
    def _create_html_template(self):
        """Create the HTML report template with embedded images and statistics."""
        # Get basic statistics
        si_stats = self._get_basic_statistics(self.silicon_data, "Silicon")
        pvk_stats = self._get_basic_statistics(self.perovskite_data, "Perovskite")
        
        # Get overlap statistics
        overlap_stats = self._calculate_overlap_statistics()
        
        # Convert images to base64 for embedding
        def image_to_base64(image_path):
            try:
                import base64
                with open(image_path, "rb") as f:
                    return base64.b64encode(f.read()).decode()
            except:
                return None
        
        # Collect all image paths
        images = {
            'silicon_calendar': self.output_dir / "calendar_heatmap_silicon.png",
            'perovskite_calendar': self.output_dir / "calendar_heatmap_perovskite.png",
            'comparison_calendar': self.output_dir / "calendar_comparison_heatmap.png",
            'temporal_overlap': self.output_dir / "temporal_overlap_analysis.png"
        }
        
        # Convert images to base64
        image_data = {}
        for name, path in images.items():
            base64_data = image_to_base64(path)
            if base64_data:
                image_data[name] = base64_data
        
        # Create HTML content
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Perovskite Data Availability Analysis - Precondition Assessment</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 20px;
            border-bottom: 3px solid #2E86AB;
        }}
        .header h1 {{
            color: #2E86AB;
            margin-bottom: 10px;
            font-size: 2.5em;
        }}
        .header p {{
            color: #666;
            font-size: 1.2em;
        }}
        .section {{
            margin: 40px 0;
            padding: 20px;
            border-left: 4px solid #2E86AB;
            background-color: #f9f9f9;
        }}
        .section h2 {{
            color: #2E86AB;
            margin-top: 0;
            font-size: 1.8em;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-left: 4px solid #2E86AB;
        }}
        .stat-card h3 {{
            color: #2E86AB;
            margin-top: 0;
            margin-bottom: 15px;
        }}
        .stat-item {{
            display: flex;
            justify-content: space-between;
            margin: 8px 0;
            padding: 5px 0;
            border-bottom: 1px solid #eee;
        }}
        .stat-label {{
            font-weight: bold;
            color: #333;
        }}
        .stat-value {{
            color: #666;
            font-family: monospace;
        }}
        .plot-container {{
            margin: 30px 0;
            text-align: center;
        }}
        .plot-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        .plot-title {{
            font-size: 1.3em;
            font-weight: bold;
            color: #2E86AB;
            margin: 20px 0 10px 0;
        }}
        .plot-description {{
            color: #666;
            font-style: italic;
            margin-bottom: 15px;
        }}
        .overlap-summary {{
            background: linear-gradient(135deg, #2E86AB, #A23B72);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
        }}
        .overlap-summary h3 {{
            margin-top: 0;
            color: white;
        }}
        .overlap-stat {{
            display: inline-block;
            margin: 10px 20px 10px 0;
            font-size: 1.1em;
        }}
        .footer {{
            text-align: center;
            margin-top: 50px;
            padding-top: 20px;
            border-top: 2px solid #ddd;
            color: #666;
        }}
        .methodology {{
            background: #f0f8ff;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .key-findings {{
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Perovskite Data Availability Analysis</h1>
            <p>Precondition Assessment for Silicon vs. Perovskite Module Comparison</p>
            <p><strong>Generated:</strong> {self.timestamp}</p>
        </div>

        <div class="section">
            <h2>Executive Summary</h2>
            <div class="key-findings">
                <h3>Key Findings</h3>
                <ul>
                    <li><strong>Silicon Modules:</strong> {si_stats['total_records']:,} total records across {si_stats['date_range_days']} days</li>
                    <li><strong>Perovskite Modules:</strong> {pvk_stats['total_records']:,} total records across {pvk_stats['date_range_days']} days</li>
                    <li><strong>Overlap Period:</strong> {overlap_stats['overlap_days']} days with data from both technologies</li>
                    <li><strong>Quality Threshold:</strong> {overlap_stats['high_quality_days']} days with ≥5 timestamps for both technologies</li>
                </ul>
            </div>
        </div>

        <div class="section">
            <h2>Basic Statistics</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <h3>Silicon Modules</h3>
                    <div class="stat-item">
                        <span class="stat-label">Total Records:</span>
                        <span class="stat-value">{si_stats['total_records']:,}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Unique Modules:</span>
                        <span class="stat-value">{si_stats['unique_modules']}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Date Range:</span>
                        <span class="stat-value">{si_stats['start_date']} to {si_stats['end_date']}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Days with Data:</span>
                        <span class="stat-value">{si_stats['date_range_days']}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Data Resolution:</span>
                        <span class="stat-value">{si_stats['resolution']}</span>
                    </div>
                </div>
                
                <div class="stat-card">
                    <h3>Perovskite Modules</h3>
                    <div class="stat-item">
                        <span class="stat-label">Total Records:</span>
                        <span class="stat-value">{pvk_stats['total_records']:,}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Unique Modules:</span>
                        <span class="stat-value">{pvk_stats['unique_modules']}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Date Range:</span>
                        <span class="stat-value">{pvk_stats['start_date']} to {pvk_stats['end_date']}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Days with Data:</span>
                        <span class="stat-value">{pvk_stats['date_range_days']}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Data Resolution:</span>
                        <span class="stat-value">{pvk_stats['resolution']}</span>
                    </div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>Calendar-Style Heatmaps</h2>
            <div class="methodology">
                <h3>Methodology</h3>
                <p>Calendar heatmaps visualize daily data availability where each day is represented as a colored square. 
                Colors indicate the number of timestamps available per day (0-24 for hourly data). This visualization 
                helps identify patterns, gaps, and seasonal variations in data collection.</p>
            </div>
"""

        # Add individual calendar heatmaps
        if 'silicon_calendar' in image_data:
            html_content += f"""
            <div class="plot-container">
                <div class="plot-title">Silicon Modules - Daily Availability Calendar</div>
                <div class="plot-description">Each square represents one day, colored by number of available timestamps (max: 24)</div>
                <img src="data:image/png;base64,{image_data['silicon_calendar']}" alt="Silicon Calendar Heatmap">
            </div>
"""

        if 'perovskite_calendar' in image_data:
            html_content += f"""
            <div class="plot-container">
                <div class="plot-title">Perovskite Modules - Daily Availability Calendar</div>
                <div class="plot-description">Each square represents one day, colored by number of available timestamps (max: 24)</div>
                <img src="data:image/png;base64,{image_data['perovskite_calendar']}" alt="Perovskite Calendar Heatmap">
            </div>
"""

        # Add comparison calendar
        if 'comparison_calendar' in image_data:
            html_content += f"""
            <div class="plot-container">
                <div class="plot-title">Technology Comparison with Overlap Analysis</div>
                <div class="plot-description">Side-by-side comparison of Silicon vs. Perovskite availability, plus overlap analysis showing days with ≥5 timestamps for both technologies</div>
                <img src="data:image/png;base64,{image_data['comparison_calendar']}" alt="Comparison Calendar Heatmap">
            </div>
"""

        html_content += """
        </div>

        <div class="section">
            <h2>Temporal Overlap Analysis</h2>
"""

        # Add overlap statistics
        html_content += f"""
            <div class="overlap-summary">
                <h3>Data Overlap Summary</h3>
                <div class="overlap-stat"><strong>Total Overlap Days:</strong> {overlap_stats['overlap_days']}</div>
                <div class="overlap-stat"><strong>High Quality Days:</strong> {overlap_stats['high_quality_days']} (≥5 timestamps both)</div>
                <div class="overlap-stat"><strong>Overlap Percentage:</strong> {overlap_stats['overlap_percentage']:.1f}%</div>
                <div class="overlap-stat"><strong>Continuous Periods:</strong> {overlap_stats['continuous_periods']}</div>
            </div>
"""

        # Add temporal overlap plot
        if 'temporal_overlap' in image_data:
            html_content += f"""
            <div class="plot-container">
                <div class="plot-title">Temporal Overlap Analysis Results</div>
                <div class="plot-description">Detailed analysis of overlapping time periods between Silicon and Perovskite data availability</div>
                <img src="data:image/png;base64,{image_data['temporal_overlap']}" alt="Temporal Overlap Analysis">
            </div>
"""

        html_content += """
        </div>

        <div class="footer">
            <p><strong>Perovskite Data Availability Analysis</strong> | Generated by ML Forecasting System</p>
            <p>This report serves as the precondition assessment for Silicon vs. Perovskite module comparison analysis.</p>
        </div>
    </div>
</body>
</html>
"""
        return html_content
    
    def _get_basic_statistics(self, data: pd.DataFrame, tech_name: str) -> Dict:
        """Get basic statistics for a technology dataset."""
        if data.empty:
            return {
                'total_records': 0,
                'unique_modules': 0,
                'start_date': 'N/A',
                'end_date': 'N/A',
                'date_range_days': 0,
                'resolution': 'N/A'
            }
        
        # Calculate statistics
        start_date = data['_time'].min().strftime('%Y-%m-%d')
        end_date = data['_time'].max().strftime('%Y-%m-%d')
        date_range_days = (data['_time'].max() - data['_time'].min()).days
        unique_modules = len(data['Name'].unique()) if 'Name' in data.columns else 'N/A'
        
        return {
            'total_records': len(data),
            'unique_modules': unique_modules,
            'start_date': start_date,
            'end_date': end_date,
            'date_range_days': date_range_days,
            'resolution': '1 hour'
        }
    
    def _calculate_overlap_statistics(self) -> Dict:
        """Calculate overlap statistics between Silicon and Perovskite data."""
        si_daily = self._get_daily_timestamp_counts(self.silicon_data, "Silicon")
        pvk_daily = self._get_daily_timestamp_counts(self.perovskite_data, "Perovskite")
        
        if si_daily.empty or pvk_daily.empty:
            return {
                'overlap_days': 0,
                'high_quality_days': 0,
                'overlap_percentage': 0.0,
                'continuous_periods': 0
            }
        
        # Find overlapping dates
        si_dates = set(si_daily.index)
        pvk_dates = set(pvk_daily.index)
        overlap_dates = si_dates.intersection(pvk_dates)
        
        # High quality days (≥5 timestamps for both)
        high_quality_days = 0
        for date in overlap_dates:
            if si_daily.get(date, 0) >= 5 and pvk_daily.get(date, 0) >= 5:
                high_quality_days += 1
        
        # Calculate overlap percentage
        total_possible_days = len(si_dates.union(pvk_dates))
        overlap_percentage = (len(overlap_dates) / total_possible_days * 100) if total_possible_days > 0 else 0
        
        # Count continuous periods (simplified)
        continuous_periods = 1 if overlap_dates else 0
        
        return {
            'overlap_days': len(overlap_dates),
            'high_quality_days': high_quality_days,
            'overlap_percentage': overlap_percentage,
            'continuous_periods': continuous_periods
        }

    def generate_summary_report(self):
        """Generate comprehensive summary report."""
        print("\nStep 5: Generating summary report...")
        
        # Create text summary
        summary_path = self.output_dir / "summary.txt"
        
        with open(summary_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("PEROVSKITE DATA AVAILABILITY ANALYSIS - PRECONDITION ASSESSMENT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Silicon Data Source: {self.silicon_path}\n")
            f.write(f"Perovskite Data Source: {self.perovskite_path}\n\n")
            
            # Basic statistics
            si_stats = self.analysis_results["basic_stats"]["silicon"]
            pvk_stats = self.analysis_results["basic_stats"]["perovskite"]
            
            f.write("DATASET OVERVIEW\n")
            f.write("-" * 40 + "\n")
            f.write(f"Silicon Modules:\n")
            f.write(f"  - Total Records: {si_stats['total_records']:,}\n")
            f.write(f"  - Unique Modules: {si_stats['unique_modules']}\n")
            f.write(f"  - Date Range: {si_stats['date_range'][0]} to {si_stats['date_range'][1]}\n")
            f.write(f"  - Resolution: {si_stats['data_resolution']}\n\n")
            
            f.write(f"Perovskite Modules:\n")
            f.write(f"  - Total Records: {pvk_stats['total_records']:,}\n")
            f.write(f"  - Unique Modules: {pvk_stats['unique_modules']}\n")
            f.write(f"  - Date Range: {pvk_stats['date_range'][0]} to {pvk_stats['date_range'][1]}\n")
            f.write(f"  - Resolution: {pvk_stats['data_resolution']}\n\n")
            
            # Temporal overlap analysis
            overlap_data = self.analysis_results["temporal_overlap"]
            f.write("TEMPORAL OVERLAP ANALYSIS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Overlap Days: {overlap_data['total_overlap_days']}\n")
            f.write(f"Silicon Only Days: {overlap_data['silicon_only_days']}\n")
            f.write(f"Perovskite Only Days: {overlap_data['perovskite_only_days']}\n")
            f.write(f"Overlap Percentage: {overlap_data['overlap_percentage']:.1f}%\n")
            
            if overlap_data["overlapping_date_range"][0]:
                f.write(f"Overlapping Period: {overlap_data['overlapping_date_range'][0]} to {overlap_data['overlapping_date_range'][1]}\n")
            
            f.write(f"\nContinuous Overlap Periods: {len(overlap_data['overlap_periods'])}\n")
            for i, period in enumerate(overlap_data["overlap_periods"]):
                f.write(f"  Period {i+1}: {period['start']} to {period['end']} ({period['duration_days']} days)\n")
            
            # Seasonal coverage
            seasonal_data = self.analysis_results["seasonal_coverage"]
            f.write("\nSEASONAL COVERAGE ANALYSIS\n")
            f.write("-" * 40 + "\n")
            for season in ["Winter", "Spring", "Summer", "Autumn"]:
                si_days = seasonal_data["silicon"][season]["unique_dates"]
                pvk_days = seasonal_data["perovskite"][season]["unique_dates"]
                f.write(f"{season:>8}: Silicon {si_days:>3} days, Perovskite {pvk_days:>3} days\n")
            
            # Module availability
            f.write(f"\nMODULE AVAILABILITY DETAILS\n")
            f.write("-" * 40 + "\n")
            
            si_modules = self.analysis_results["module_availability"]["silicon"]
            pvk_modules = self.analysis_results["module_availability"]["perovskite"]
            
            f.write(f"Silicon Modules ({si_modules['module_count']} total):\n")
            if "module_details" in si_modules:
                for module in si_modules["module_details"][:10]:  # Limit to first 10
                    f.write(f"  {module['name']}: {module['active_days']} active days\n")
                if len(si_modules["module_details"]) > 10:
                    f.write(f"  ... and {len(si_modules['module_details']) - 10} more modules\n")
            
            f.write(f"\nPerovskite Modules ({pvk_modules['module_count']} total):\n")
            if "module_details" in pvk_modules:
                for module in pvk_modules["module_details"][:10]:  # Limit to first 10
                    f.write(f"  {module['name']}: {module['active_days']} active days\n")
                if len(pvk_modules["module_details"]) > 10:
                    f.write(f"  ... and {len(pvk_modules['module_details']) - 10} more modules\n")
            
            # Recommendations
            f.write(f"\nRECOMMENDations FOR MODEL COMPARISON\n")
            f.write("-" * 40 + "\n")
            
            if overlap_data["total_overlap_days"] > 30:
                f.write("[OK] Sufficient temporal overlap for fair model comparison\n")
            else:
                f.write("[WARN] Limited temporal overlap - consider extending data collection period\n")
            
            min_seasonal_days = min([
                seasonal_data["silicon"][s]["unique_dates"] + seasonal_data["perovskite"][s]["unique_dates"]
                for s in ["Winter", "Spring", "Summer", "Autumn"]
            ])
            
            if min_seasonal_days > 20:
                f.write("[OK] Balanced seasonal coverage across all seasons\n")
            else:
                f.write("[WARN] Unbalanced seasonal coverage - some seasons underrepresented\n")
            
            if si_modules["module_count"] >= 3 and pvk_modules["module_count"] >= 3:
                f.write("[OK] Sufficient module diversity for robust comparison\n")
            else:
                f.write("[WARN] Limited module diversity - results may not be generalizable\n")
            
            f.write(f"\nFILES GENERATED\n")
            f.write("-" * 40 + "\n")
            f.write(f"- summary.txt (this file)\n")
            f.write(f"- data_availability_heatmaps.png\n")
            f.write(f"- temporal_overlap_analysis.png\n")
            f.write(f"- interactive_availability_analysis.html\n")
        
        print(f"OK Summary report saved: {summary_path}")
    
    def run_complete_analysis(self) -> bool:
        """
        Run the complete precondition analysis workflow.
        
        Returns:
            bool: True if analysis completed successfully
        """
        try:
            # Load data
            if not self.load_data():
                return False
            
            # Perform analysis
            self.analyze_data_availability()
            
            # Create visualizations
            self.create_heatmaps()
            self.create_overlap_analysis_plots()
            
            # Generate reports
            self.generate_html_report()
            self.generate_summary_report()
            
            print(f"\n{'='*60}")
            print("[OK] PRECONDITION ANALYSIS COMPLETED SUCCESSFULLY")
            print(f"{'='*60}")
            print(f"Results saved to: {self.output_dir}")
            
            return True
            
        except Exception as e:
            print(f"\n{'='*60}")
            print(f"[FAIL] ANALYSIS FAILED: {e}")
            print(f"{'='*60}")
            return False


def main():
    """Main function to run the precondition analysis."""
    parser = argparse.ArgumentParser(
        description="Perovskite Data Availability Analysis - Precondition Assessment"
    )
    parser.add_argument(
        "--silicon-path", 
        required=True,
        help="Path to Silicon clean data CSV file"
    )
    parser.add_argument(
        "--perovskite-path", 
        required=True,
        help="Path to Perovskite clean data CSV file"
    )
    parser.add_argument(
        "--output-dir", 
        help="Base output directory (optional)"
    )
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = PerovskiteDataAvailabilityAnalyzer(
        silicon_path=args.silicon_path,
        perovskite_path=args.perovskite_path,
        output_base_dir=args.output_dir
    )
    
    # Run analysis
    success = analyzer.run_complete_analysis()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
