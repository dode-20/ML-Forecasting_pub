import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from typing import Optional, Tuple, List


class WeatherDataMerger:
    """
    Class to merge weather data with clean PV data
    
    This class handles the integration of weather parameters into existing clean PV data files,
    ensuring proper timestamp matching and data validation. Only merges weather features
    that are specified in the config file.
    """
    
    def __init__(self, weather_data_path: str, config_path: Optional[str] = None):
        """
        Initialize WeatherDataMerger
        
        Args:
            weather_data_path: Path to the weather data CSV file (interpolated 5-min data)
            config_path: Path to the model config file to determine which weather features to merge
        """
        self.weather_data_path = weather_data_path
        self.config_path = config_path
        self.weather_df = None
        self.required_weather_columns = ['TT_10', 'RF_10', 'V_N', 'RWS_10', 'RWS_IND_10', 'GS_10', 'SD_10']
        self.selected_weather_features = []
        
    def load_config(self) -> bool:
        """
        Load config file to determine which weather features to merge
        
        Returns:
            bool: True if successful, False otherwise
        """
        if self.config_path is None:
            print("WARNING: No config file provided, will merge all available weather features")
            self.selected_weather_features = self.required_weather_columns
            return True
            
        try:
            print(f"Loading config from: {self.config_path}")
            
            if not os.path.exists(self.config_path):
                print(f"ERROR: Config file not found: {self.config_path}")
                return False
            
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            # Extract weather features from config
            weather_config = config.get("weather_data", {})
            if weather_config.get("use_weatherData", False):
                self.selected_weather_features = weather_config.get("weather_features", [])
                print(f"SUCCESS: Loaded weather features from config: {self.selected_weather_features}")
            else:
                print("WARNING: Weather data disabled in config, will merge all available features")
                self.selected_weather_features = self.required_weather_columns
            
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to load config: {e}")
            print("WARNING: Will merge all available weather features")
            self.selected_weather_features = self.required_weather_columns
            return True
    
    def load_weather_data(self) -> bool:
        """
        Load weather data from CSV file
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print(f"Loading weather data from: {self.weather_data_path}")
            
            if not os.path.exists(self.weather_data_path):
                print(f"ERROR: Weather data file not found: {self.weather_data_path}")
                return False
            
            # Load weather data
            self.weather_df = pd.read_csv(self.weather_data_path)
            
            # Convert timestamp to datetime
            self.weather_df['timestamp'] = pd.to_datetime(self.weather_df['timestamp'])
            
            # Set timestamp as index for efficient merging
            self.weather_df.set_index('timestamp', inplace=True)
            
            # Validate required columns
            missing_columns = [col for col in self.required_weather_columns 
                             if col not in self.weather_df.columns]
            
            if missing_columns:
                print(f"ERROR: Missing required weather columns: {missing_columns}")
                print(f"Available columns: {list(self.weather_df.columns)}")
                return False
            
            print(f"SUCCESS: Loaded weather data with {len(self.weather_df)} records")
            print(f"   Time range: {self.weather_df.index.min()} to {self.weather_df.index.max()}")
            print(f"   Columns: {list(self.weather_df.columns)}")
            
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to load weather data: {e}")
            return False
    
    def load_clean_data(self, clean_data_path: str) -> Optional[pd.DataFrame]:
        """
        Load clean PV data from CSV file
        
        Args:
            clean_data_path: Path to the clean PV data CSV file
            
        Returns:
            pd.DataFrame: Clean data DataFrame or None if failed
        """
        try:
            print(f"Loading clean data from: {clean_data_path}")
            
            if not os.path.exists(clean_data_path):
                print(f"ERROR: Clean data file not found: {clean_data_path}")
                return None
            
            # Load clean data
            clean_df = pd.read_csv(clean_data_path)
            
            # Convert timestamp to datetime (handle both 'timestamp' and '_time' column names)
            if 'timestamp' in clean_df.columns:
                timestamp_col = 'timestamp'
            elif '_time' in clean_df.columns:
                timestamp_col = '_time'
                # Rename _time to timestamp for consistency
                clean_df = clean_df.rename(columns={'_time': 'timestamp'})
            else:
                print(f"ERROR: No timestamp column found. Available columns: {list(clean_df.columns)}")
                return None
            
            clean_df['timestamp'] = pd.to_datetime(clean_df['timestamp'])
            
            print(f"SUCCESS: Loaded clean data with {len(clean_df)} records")
            print(f"   Time range: {clean_df['timestamp'].min()} to {clean_df['timestamp'].max()}")
            print(f"   Columns: {list(clean_df.columns)}")
            
            return clean_df
            
        except Exception as e:
            print(f"ERROR: Failed to load clean data: {e}")
            return None
    
    def validate_time_coverage(self, clean_df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Validate if weather data covers the entire clean data time range
        
        Args:
            clean_df: Clean data DataFrame
            
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        clean_start = clean_df['timestamp'].min()
        clean_end = clean_df['timestamp'].max()
        
        if self.weather_df is None:
            return False, "Weather data not loaded"
            
        weather_start = self.weather_df.index.min()
        weather_end = self.weather_df.index.max()
        
        print(f"\nTime range validation:")
        print(f"   Clean data: {clean_start} to {clean_end}")
        print(f"   Weather data: {weather_start} to {weather_end}")
        
        # Check if weather data covers clean data range
        if weather_start > clean_start:
            error_msg = f"Weather data starts too late: {weather_start} > {clean_start}"
            return False, error_msg
        
        if weather_end < clean_end:
            error_msg = f"Weather data ends too early: {weather_end} < {clean_end}"
            return False, error_msg
        
        print(f"SUCCESS: Weather data fully covers clean data time range")
        return True, ""
    
    def merge_data(self, clean_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Merge weather data with clean PV data
        
        Args:
            clean_df: Clean data DataFrame
            
        Returns:
            pd.DataFrame: Merged DataFrame or None if failed
        """
        try:
            print(f"\nMerging weather data with clean data...")
            print(f"Selected weather features: {self.selected_weather_features}")
            
            # Set timestamp as index for merging
            clean_df_indexed = clean_df.set_index('timestamp')
            
            # Check if weather data is loaded
            if self.weather_df is None:
                print(f"ERROR: Weather data not loaded")
                return None
            
            # Check which selected features are available in weather data
            available_features = [f for f in self.selected_weather_features if f in self.weather_df.columns]
            missing_features = [f for f in self.selected_weather_features if f not in self.weather_df.columns]
            
            if missing_features:
                print(f"WARNING: Some selected weather features are not available in weather data: {missing_features}")
                print(f"Available weather columns: {list(self.weather_df.columns)}")
            
            if not available_features:
                print(f"ERROR: No selected weather features are available in weather data")
                return None
            
            print(f"Will merge available features: {available_features}")

            # Merge weather data with clean data (only selected features)
            merged_df = clean_df_indexed.merge(
                self.weather_df[available_features],
                left_index=True,
                right_index=True,
                how='left'
            )
            
            # Reset index to get timestamp back as column
            merged_df.reset_index(inplace=True)
            
            # Check for missing values
            missing_counts = merged_df[available_features].isna().sum()
            total_records = len(merged_df)
            
            print(f"Merge results:")
            print(f"   Total records: {total_records}")
            for col in available_features:
                missing = missing_counts[col]
                percentage = (missing / total_records) * 100
                print(f"   {col}: {missing} missing ({percentage:.1f}%)")
            
            # Check if any weather column has missing values
            if missing_counts.sum() > 0:
                print(f"WARNING: Some weather data is missing")
                print(f"   This may indicate timestamp mismatches or data gaps")
            
            print(f"SUCCESS: Data merged successfully")
            return merged_df
            
        except Exception as e:
            print(f"ERROR: Failed to merge data: {e}")
            return None
    
    def save_merged_data(self, merged_df: pd.DataFrame, original_path: str) -> bool:
        """
        Save merged data to CSV file
        
        Args:
            merged_df: Merged DataFrame
            original_path: Original clean data file path
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create output path
            base_path = os.path.splitext(original_path)[0]
            output_path = f"{base_path}_weather-integrated.csv"
            
            # Save merged data
            merged_df.to_csv(output_path, index=False)
            
            print(f"SUCCESS: Merged data saved to: {output_path}")
            print(f"   File size: {os.path.getsize(output_path) / 1024:.1f} KB")
            
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to save merged data: {e}")
            return False
    
    def process_clean_data(self, clean_data_path: str) -> bool:
        """
        Complete process to merge weather data with clean PV data
        
        Args:
            clean_data_path: Path to the clean PV data CSV file
            
        Returns:
            bool: True if successful, False otherwise
        """
        print("="*60)
        print("WEATHER DATA MERGER")
        print("="*60)
        
        # Step 1: Load config to determine weather features
        if not self.load_config():
            return False
        
        # Step 2: Load weather data
        if not self.load_weather_data():
            return False
        
        # Step 3: Load clean data
        clean_df = self.load_clean_data(clean_data_path)
        if clean_df is None:
            return False
        
        # Step 4: Validate time coverage
        is_valid, error_msg = self.validate_time_coverage(clean_df)
        if not is_valid:
            print(f"ERROR: Time coverage validation failed: {error_msg}")
            return False
        
        # Step 5: Merge data
        merged_df = self.merge_data(clean_df)
        if merged_df is None:
            return False
        
        # Step 6: Save merged data
        if not self.save_merged_data(merged_df, clean_data_path):
            return False
        
        # Step 6.5: Fill missing timestamps with zeros for complete time series
        print("Step 6.5: Filling missing timestamps with zeros...")
        # Determine resolution from data (default to 5min if can't determine)
        time_diff = pd.to_datetime(merged_df['timestamp']).diff().median()
        if pd.isna(time_diff):
            resolution = "5min"
        elif time_diff <= pd.Timedelta(minutes=5):
            resolution = "5min"
        elif time_diff <= pd.Timedelta(minutes=10):
            resolution = "10min"
        elif time_diff <= pd.Timedelta(hours=1):
            resolution = "1h"
        else:
            resolution = "5min"
        
        print(f"Detected resolution: {resolution}")
        filled_df = self._fill_missing_timestamps_with_zeros(merged_df, resolution)
        
        # Save the final zero-filled data
        if not self.save_merged_data(filled_df, clean_data_path):
            print("WARNING: Failed to save zero-filled data, but original merged data was saved")
        
        print("="*60)
        print("MERGE PROCESS COMPLETED SUCCESSFULLY")
        print("="*60)
        
        return True

    # --- New helpers for multi-resolution processing ---
    def process_clean_data_for_resolution(self, clean_data_path: str, resolution: str, config_path: Optional[str] = None) -> bool:
        """
        Process one clean-data CSV for a given resolution by selecting the matching
        weather CSV of the same resolution and producing an integrated output.

        - clean_data_path: e.g. results/training_data/cleanData/.../_clean-10min.csv
        - resolution: one of {"5min","10min","1h"}
        - config_path: optional model config path to select weather features
        """
        # Map resolution to weather file
        weather_map = {
            "5min": "results/weather_data/cleanData/combined_weatherData-5min.csv",
            "10min": "results/weather_data/cleanData/combined_weatherData-10min.csv",
            "1h": "results/weather_data/cleanData/combined_weatherData-1h.csv",
        }
        if resolution not in weather_map:
            print(f"ERROR: Unsupported resolution: {resolution}")
            return False

        self.weather_data_path = weather_map[resolution]
        if config_path is not None:
            self.config_path = config_path

        print("=" * 60)
        print(f"WEATHER DATA MERGER ({resolution})")
        print("=" * 60)

        # Load config and weather
        if not self.load_config():
            return False
        if not self.load_weather_data():
            return False

        # Load clean PV data (matching resolution)
        clean_df = self.load_clean_data(clean_data_path)
        if clean_df is None:
            return False

        # Validate coverage and merge
        is_valid, error_msg = self.validate_time_coverage(clean_df)
        if not is_valid:
            print(f"ERROR: Time coverage validation failed: {error_msg}")
            return False

        merged_df = self.merge_data(clean_df)
        if merged_df is None:
            return False

        # Step 6.5: Fill missing timestamps with zeros for complete time series
        print("Step 6.5: Filling missing timestamps with zeros...")
        print(f"Using configured resolution: {resolution}")
        filled_df = self._fill_missing_timestamps_with_zeros(merged_df, resolution)
        
        # Save the final zero-filled data
        if not self.save_merged_data(filled_df, clean_data_path):
            print("WARNING: Failed to save zero-filled data, but original merged data was saved")
            return False
        
        return True

    def process_all_resolutions(self, clean_base_path: str, config_path: Optional[str] = None) -> bool:
        """
        Process all three resolutions by deriving paths from a base clean path.

        clean_base_path should be the base without resolution suffix, e.g.
        results/training_data/cleanData/.../_clean (without -5min.csv).
        The method will try _clean-5min.csv, _clean-10min.csv, _clean-1h.csv
        if they exist, and produce matching integrated outputs.
        """
        success_all = True
        for res in ["5min", "10min", "1h"]:
            candidate = f"{clean_base_path}-{res}.csv"
            if os.path.exists(candidate):
                ok = self.process_clean_data_for_resolution(candidate, res, config_path)
                success_all = success_all and ok
            else:
                print(f"[WARN] Clean file for {res} not found: {candidate}")
        return success_all

    def _fill_missing_timestamps_with_zeros(self, data: pd.DataFrame, resolution: str = "5min") -> pd.DataFrame:
        """
        Fill missing timestamps with zero values to ensure complete time series.
        Only sets P and Irr to zero, weather features get values from weather data file.
        
        Args:
            data (pd.DataFrame): Input data with timestamp column 'timestamp'
            resolution (str): Data resolution ('5min', '10min', '1h')
            
        Returns:
            pd.DataFrame: Data with all missing timestamps filled with zeros for PV features only
        """
        print(f"Step 6.5: Filling missing timestamps with zeros for resolution: {resolution}")
        
        if 'timestamp' not in data.columns:
            print("WARNING: No 'timestamp' column found, skipping timestamp filling")
            return data
        
        # Load weather data from file to get all timestamps
        if not hasattr(self, 'weather_df') or self.weather_df is None:
            print("WARNING: Weather data not loaded, cannot fill weather features")
            return data
        
        print(f"Using weather data from: {self.weather_data_path}")
        
        # Convert timestamp to datetime if needed
        if data['timestamp'].dtype == 'object':
            data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Get time range from data
        start_time = data['timestamp'].min()
        end_time = data['timestamp'].max()
        
        print(f"Data time range: {start_time} to {end_time}")
        
        # Create complete timestamp range based on resolution
        if resolution == "5min":
            freq = "5min"
        elif resolution == "10min":
            freq = "10min"
        elif resolution == "1h":
            freq = "H"
        else:
            print(f"WARNING: Unknown resolution {resolution}, using 5min as default")
            freq = "5min"
        
        # Generate complete timestamp range
        complete_timestamps = pd.date_range(start=start_time, end=end_time, freq=freq)
        print(f"Generated {len(complete_timestamps)} complete timestamps")
        
        # Create zero-filled dataset for missing timestamps
        zero_data = []
        # Convert freq string to pandas offset for rounding
        if freq == "5min":
            freq_offset = pd.Timedelta(minutes=5)
        elif freq == "10min":
            freq_offset = pd.Timedelta(minutes=10)
        elif freq == "H":
            freq_offset = pd.Timedelta(hours=1)
        else:
            freq_offset = pd.Timedelta(minutes=5)
        
        existing_timestamps = set(data['timestamp'].dt.round(freq_offset))
        
        # Define which columns should get zero values (only PV features)
        zero_columns = ['P', 'Irr']
        
        # Get weather columns for copying values
        weather_columns = ['TT_10', 'RF_10']
        
        for timestamp in complete_timestamps:
            if timestamp not in existing_timestamps:
                # Create zero row for this timestamp
                zero_row = {}
                
                # Add values for all columns
                for col in data.columns:
                    if col == 'timestamp':
                        zero_row[col] = timestamp
                    elif col == 'module_type':
                        # Use 'unknown' for module type
                        zero_row[col] = 'unknown'
                    elif col in ['day_of_year', 'month', 'weekday', 'hour', 'minute']:
                        # Time features should have proper values from timestamp
                        if col == 'day_of_year':
                            zero_row[col] = timestamp.dayofyear
                        elif col == 'month':
                            zero_row[col] = timestamp.month
                        elif col == 'weekday':
                            zero_row[col] = timestamp.weekday()
                        elif col == 'hour':
                            zero_row[col] = timestamp.hour
                        elif col == 'minute':
                            zero_row[col] = timestamp.minute
                    elif col in zero_columns:
                        # PV features get zero
                        zero_row[col] = 0.0
                    elif col in weather_columns:
                        # Weather features get values from weather data file
                        # Find the exact timestamp in weather data
                        if timestamp in self.weather_df.index:
                            zero_row[col] = self.weather_df.loc[timestamp, col]
                        else:
                            # If exact timestamp not found, find closest
                            time_diffs = abs(self.weather_df.index - timestamp)
                            closest_idx = time_diffs.argmin()
                            zero_row[col] = self.weather_df.iloc[closest_idx][col]
                    else:
                        # Any other columns get zero as well
                        zero_row[col] = 0.0
                
                zero_data.append(zero_row)
        
        if zero_data:
            print(f"Adding {len(zero_data)} zero-filled rows for missing timestamps")
            print(f"Columns set to zero: {zero_columns}")
            print(f"Weather features copied from weather data file: {weather_columns}")
            zero_df = pd.DataFrame(zero_data)
            
            # Combine original data with zero-filled data
            combined_data = pd.concat([data, zero_df], ignore_index=True)
            
            # Sort by timestamp
            combined_data = combined_data.sort_values('timestamp').reset_index(drop=True)
            
            print(f"Final data shape after zero-filling: {combined_data.shape}")
            return combined_data
        else:
            print("No missing timestamps found, data is already complete")
            return data


def main():
    """
    Example: merge all three resolutions for a given clean dataset directory.
    Adjust base paths as needed.
    """
    import sys
    from pathlib import Path
    
    # Get config path
    config_path = "results/model_configs/test_lstm_model_settings.json"
    
    # Load config to determine module type and date range
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Extract module type and date range from config
        module_type = config.get('module_type', 'silicon').capitalize()
        start_date = config.get('date_selection', {}).get('start', '2024-08-17').replace('-', '')
        end_date = config.get('date_selection', {}).get('end', '2025-06-22').replace('-', '')
        model_name = config.get('model_name', 'test_lstm_model')
        date_range = f"{start_date}_{end_date}"
        
        # Build correct clean data base path
        clean_base = f"results/training_data/{module_type}/cleanData/{date_range}/{date_range}_{model_name}_clean"
        
        print(f"Using config-based paths:")
        print(f"  Module type: {module_type}")
        print(f"  Date range: {date_range}")
        print(f"  Clean base path: {clean_base}")
        
    except Exception as e:
        print(f"ERROR: Failed to load config: {e}")
        print("Using fallback paths...")
        # Fallback to Silicon with current date range
        clean_base = "results/training_data/Silicon/cleanData/20240901_20250725/20240901_20250725_test_lstm_model_clean"
        config_path = "results/model_configs/test_lstm_model_settings.json"

    merger = WeatherDataMerger(weather_data_path="", config_path=config_path)
    success = merger.process_all_resolutions(clean_base, config_path=config_path)

    if success:
        print("Weather data integration for all resolutions completed successfully!")
    else:
        print("Weather data integration had errors. Check logs above.")


if __name__ == "__main__":
    main() 