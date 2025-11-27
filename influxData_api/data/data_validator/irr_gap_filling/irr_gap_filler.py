"""
Irr Gap Filler Module

This module provides functionality for filling missing irradiance values
using GS_10 weather data and scaling factors.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger('influxData_api.data_validator_new')


class IrrGapFiller:
    """
    Class for filling missing irradiance values using GS_10 data and scaling factors.
    """
    
    def __init__(self):
        """Initialize the IrrGapFiller."""
        pass
    
    def _fill_from_cross_module_irr(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing Irr values using cross-module Irr data (Irr_si or Irr_pvk).
        
        Args:
            data (pd.DataFrame): Input data with potentially missing Irr values
            
        Returns:
            pd.DataFrame: Data with Irr values filled from cross-module data
        """
        # Check if cross-module Irr columns exist
        cross_module_columns = ['Irr_si', 'Irr_pvk']
        available_cross_columns = [col for col in cross_module_columns if col in data.columns]
        
        if not available_cross_columns:
            logger.debug("No cross-module Irr columns found, skipping cross-module filling")
            return data
        
        missing_irr_mask = data['Irr'].isna()
        filled_count = 0
        
        for idx in data[missing_irr_mask].index:
            # Try each available cross-module column
            for cross_col in available_cross_columns:
                cross_value = data.loc[idx, cross_col]
                
                # Check if cross-module value is available and not NaN
                if pd.notna(cross_value) and cross_value > 0:
                    data.loc[idx, 'Irr'] = cross_value
                    filled_count += 1
                    
                    # Debug output for first 10 filled values
                    if filled_count <= 10:
                        timestamp = data.loc[idx, '_time'] if '_time' in data.columns else f"Row {idx}"
                        logger.info(f"DEBUG: Cross-module fill #{filled_count}: {timestamp} -> Irr filled with {cross_col}={cross_value:.2f}")
                    
                    break  # Stop after first successful fill
        
        if filled_count > 0:
            logger.info(f"Cross-module filling completed: {filled_count} Irr values filled")
        
        return data
    
    def fill_missing_irr_with_gs10(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing Irr values using GS_10 data and scaling factors.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Data with filled Irr values
        """
        logger.info("Starting Irr value filling using GS_10 data and scaling factors...")
        
        # Check if Irr column exists and has missing values
        if 'Irr' not in data.columns:
            logger.info("No Irr column found, skipping Irr filling")
            return data
        
        missing_irr_mask = data['Irr'].isna()
        total_missing = missing_irr_mask.sum()
        
        if total_missing == 0:
            logger.info("No missing Irr values found, skipping Irr filling")
            return data
        
        logger.info(f"Found {total_missing} missing Irr values to fill")
        
        # FIRST PRIORITY: Try to fill missing Irr values using cross-module Irr (Irr_si or Irr_pvk)
        filled_data = data.copy()
        cross_module_filled = self._fill_from_cross_module_irr(filled_data)
        
        # Recalculate missing values after cross-module filling
        remaining_missing_mask = filled_data['Irr'].isna()
        remaining_missing = remaining_missing_mask.sum()
        cross_filled_count = total_missing - remaining_missing
        
        if cross_filled_count > 0:
            logger.info(f"Filled {cross_filled_count} missing Irr values using cross-module data")
        
        if remaining_missing == 0:
            logger.info("All missing Irr values filled using cross-module data")
            return filled_data
        
        logger.info(f"Remaining {remaining_missing} missing Irr values to fill using GS_10 data")
        
        # Check if we have the required time features for GS_10 filling
        required_time_features = ['day_of_year', 'hour']
        missing_time_features = [feat for feat in required_time_features if feat not in data.columns]
        
        if missing_time_features:
            logger.warning(f"Missing required time features: {missing_time_features}. Cannot fill remaining Irr values.")
            return filled_data
        
        # Load scaling factors from irr comparison
        scaling_factors = self._load_scaling_factors()
        if scaling_factors is None:
            logger.warning("Could not load scaling factors, skipping Irr filling")
            return data
        
        # Load GS_10 data
        gs10_data = self._load_gs10_data()
        if gs10_data is None:
            logger.warning("Could not load GS_10 data, skipping Irr filling")
            return data
        
        # Continue with GS_10 filling for remaining missing values
        # filled_data already created above
        
        # Fill remaining missing Irr values using GS_10
        filled_count = 0
        removed_count = 0
        removed_reasons = {
            'no_scaling_factor_anytime': 0,
            'gs10_zero': 0,
            'nighttime': 0,
            'other': 0
        }
        
        # Track rows to remove
        rows_to_remove = []
        
        for idx in filled_data[remaining_missing_mask].index:
            row = data.loc[idx]
            day_of_year = row['day_of_year']
            hour = row['hour']
            
            # Determine daytime category
            daytime = self._classify_daytime(hour)
            
            # Skip nighttime hours (no solar radiation) - mark for removal
            if daytime == 'night':
                removed_reasons['nighttime'] += 1
                rows_to_remove.append(idx)
                continue
            
            # Try to get scaling factor with fallback logic
            scaling_factor = self._get_scaling_factor_with_fallback(scaling_factors, day_of_year, daytime)
            
            if scaling_factor is None:
                # No scaling factor available for this day at all - mark for removal
                removed_reasons['no_scaling_factor_anytime'] += 1
                rows_to_remove.append(idx)
                continue
            
            # Get GS_10 value for this timestamp
            gs10_value = self._get_gs10_value(gs10_data, row['_time'])
            
            if gs10_value is None or gs10_value <= 0:
                # No GS_10 value or zero value - mark for removal
                removed_reasons['gs10_zero'] += 1
                rows_to_remove.append(idx)
                continue
            
            # Calculate Irr value: Irr = GS_10 * scaling_factor
            irr_value = gs10_value * scaling_factor
            filled_data.loc[idx, 'Irr'] = irr_value
            filled_count += 1
        
        # Remove all problematic rows
        if rows_to_remove:
            filled_data = filled_data.drop(rows_to_remove)
            removed_count = len(rows_to_remove)
            logger.info(f"Removed {removed_count} problematic rows to ensure clean data")
        
        # Log detailed statistics
        logger.info(f"Successfully filled {filled_count} missing Irr values")
        logger.info(f"Removed {removed_count} problematic rows:")
        for reason, count in removed_reasons.items():
            if count > 0:
                logger.info(f"  - {reason}: {count} values")
        
        # Log some examples of removed values for debugging
        if removed_count > 0:
            logger.info("Sample of removed values for debugging:")
            sample_removed = data.loc[rows_to_remove[:10]] if len(rows_to_remove) >= 10 else data.loc[rows_to_remove]
            for idx, row in sample_removed.iterrows():
                daytime = self._classify_daytime(row['hour'])
                scaling_factor = self._get_scaling_factor_with_fallback(scaling_factors, row['day_of_year'], daytime)
                gs10_value = self._get_gs10_value(gs10_data, row['_time'])
                
                logger.info(f"  Row {idx}: day={row['day_of_year']}, hour={row['hour']}, "
                           f"daytime={daytime}, scaling_factor={scaling_factor}, gs10_value={gs10_value}")
        
        return filled_data
    
    def _load_scaling_factors(self) -> Optional[pd.DataFrame]:
        """
        Load scaling factors from the irr comparison analysis.
        
        Returns:
            Optional[pd.DataFrame]: Scaling factors data or None if not found
        """
        try:
            # Look for the most recent daily_scaling_factors.csv file
            irr_comparison_dir = Path("results/irr_comparison")
            if not irr_comparison_dir.exists():
                logger.warning("irr_comparison directory not found")
                return None
            
            # Find the most recent subdirectory
            subdirs = [d for d in irr_comparison_dir.iterdir() if d.is_dir()]
            if not subdirs:
                logger.warning("No subdirectories found in irr_comparison")
                return None
            
            # Sort by creation time and get the most recent
            latest_dir = max(subdirs, key=lambda x: x.stat().st_mtime)
            scaling_factors_path = latest_dir / "daily_scaling_factors.csv"
            
            if not scaling_factors_path.exists():
                logger.warning(f"daily_scaling_factors.csv not found in {latest_dir}")
                return None
            
            scaling_factors = pd.read_csv(scaling_factors_path)
            logger.info(f"Loaded scaling factors from {scaling_factors_path}")
            return scaling_factors
            
        except Exception as e:
            logger.error(f"Error loading scaling factors: {e}")
            return None
    
    def _load_gs10_data(self) -> Optional[pd.DataFrame]:
        """
        Load GS_10 data from weather data CSV files.
        
        Returns:
            Optional[pd.DataFrame]: GS_10 data or None if not found
        """
        try:
            # Try to find the appropriate GS_10 data file based on data resolution
            weather_data_dir = Path("results/weather_data/cleanData")
            if not weather_data_dir.exists():
                logger.warning("weather_data directory not found")
                return None
            
            # Look for GS_10 data files with different resolutions
            gs10_files = [
                "combined_weatherData-5min.csv",
                "combined_weatherData-10min.csv", 
                "combined_weatherData-1h.csv"
            ]
            
            gs10_data = None
            for filename in gs10_files:
                file_path = weather_data_dir / filename
                if file_path.exists():
                    logger.info(f"Loading GS_10 data from {file_path}")
                    gs10_data = pd.read_csv(file_path)
                    # Keep only timestamp and GS_10 columns
                    if 'timestamp' in gs10_data.columns and 'GS_10' in gs10_data.columns:
                        gs10_data = gs10_data[['timestamp', 'GS_10']].copy()
                        gs10_data['timestamp'] = pd.to_datetime(gs10_data['timestamp'])
                        gs10_data = gs10_data.dropna(subset=['GS_10'])
                        break
            
            if gs10_data is None:
                logger.warning("No suitable GS_10 data file found")
                return None
            
            logger.info(f"Loaded GS_10 data with {len(gs10_data)} records")
            return gs10_data
            
        except Exception as e:
            logger.error(f"Error loading GS_10 data: {e}")
            return None
    
    def _classify_daytime(self, hour: int) -> str:
        """
        Classify hour into daytime category.
        
        Args:
            hour (int): Hour of day (0-23)
            
        Returns:
            str: Daytime category (morning, midday, evening, night)
        """
        if 5 <= hour <= 10:
            return 'morning'
        elif 11 <= hour <= 16:
            return 'midday'
        elif 17 <= hour <= 21:
            return 'evening'
        else:
            return 'night'
    
    def _get_scaling_factor(self, scaling_factors: pd.DataFrame, day_of_year: int, daytime: str) -> Optional[float]:
        """
        Get scaling factor for a specific day and daytime.
        
        Args:
            scaling_factors (pd.DataFrame): Scaling factors data
            day_of_year (int): Day of year
            daytime (str): Daytime category
            
        Returns:
            Optional[float]: Scaling factor or None if not found
        """
        try:
            # Filter scaling factors for this day and daytime
            mask = (scaling_factors['day_of_year'] == day_of_year) & (scaling_factors['daytime'] == daytime)
            filtered_factors = scaling_factors[mask]
            
            if filtered_factors.empty:
                return None
            
            # Use the median scaling factor for this day and daytime
            scaling_factor = filtered_factors['Median'].iloc[0]
            return scaling_factor
            
        except Exception as e:
            logger.error(f"Error getting scaling factor for day {day_of_year}, {daytime}: {e}")
            return None
    
    def _get_gs10_value(self, gs10_data: pd.DataFrame, timestamp: pd.Timestamp) -> Optional[float]:
        """
        Get GS_10 value for a specific timestamp.
        
        Args:
            gs10_data (pd.DataFrame): GS_10 data
            timestamp (pd.Timestamp): Target timestamp
            
        Returns:
            Optional[float]: GS_10 value or None if not found
        """
        try:
            # Round timestamp to nearest minute for matching
            rounded_timestamp = timestamp.round('min')
            
            # Find the closest timestamp in GS_10 data
            gs10_data['timestamp_rounded'] = gs10_data['timestamp'].round('min')
            
            # Try exact match first
            exact_match = gs10_data[gs10_data['timestamp_rounded'] == rounded_timestamp]
            if not exact_match.empty:
                return exact_match['GS_10'].iloc[0]
            
            # If no exact match, find the closest timestamp within 5 minutes
            time_diff = abs(gs10_data['timestamp_rounded'] - rounded_timestamp)
            closest_idx = time_diff.idxmin()
            
            if time_diff[closest_idx] <= pd.Timedelta(minutes=5):
                return gs10_data.loc[closest_idx, 'GS_10']
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting GS_10 value for timestamp {timestamp}: {e}")
            return None

    def _get_scaling_factor_with_fallback(self, scaling_factors: pd.DataFrame, day_of_year: int, daytime: str) -> Optional[float]:
        """
        Get scaling factor for a specific day and daytime with fallback logic.
        
        Args:
            scaling_factors (pd.DataFrame): Scaling factors data
            day_of_year (int): Day of year
            daytime (str): Preferred daytime category
            
        Returns:
            Optional[float]: Scaling factor or None if none available for the entire day
        """
        try:
            # First try to get the preferred daytime scaling factor
            scaling_factor = self._get_scaling_factor(scaling_factors, day_of_year, daytime)
            if scaling_factor is not None:
                return scaling_factor
            
            # If preferred daytime not available, try fallback options
            fallback_order = ['midday', 'morning', 'evening']  # Most reliable to least reliable
            
            # Remove the preferred daytime from fallback order if it's already there
            if daytime in fallback_order:
                fallback_order.remove(daytime)
            
            # Try fallback options
            for fallback_daytime in fallback_order:
                scaling_factor = self._get_scaling_factor(scaling_factors, day_of_year, fallback_daytime)
                if scaling_factor is not None:
                    logger.debug(f"Using fallback scaling factor for day {day_of_year}: {fallback_daytime} instead of {daytime}")
                    return scaling_factor
            
            # No scaling factor available for this day at all
            logger.debug(f"No scaling factor available for day {day_of_year} for any daytime period")
            return None
            
        except Exception as e:
            logger.error(f"Error getting scaling factor with fallback for day {day_of_year}, {daytime}: {e}")
            return None
