#!/usr/bin/env python3
"""
Run script for the new modular DataValidator implementation.

Validates the training data from the combined CSV and saves the results.
Supports configurable paths and model settings.
Uses the new modular structure but maintains the same interface as the original.
"""

import sys
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import re
from typing import Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

# Import the new modular DataValidator
from influxData_api.data.data_validator import DataValidator


class RunDataValidatorNew:
    def __init__(self, config_file: str = "test_lstm_model_settings.json"):
        """
        Initialize the data validator run with configuration
        
        Args:
            config_file: Path to JSON configuration file (full path or filename)
        """
        self.config_file = config_file
        self.config = self.load_config()
        
        # Determine paths based on config
        self.setup_paths()
        
        # Get module_type for display
        module_type = self.config.get('module_type', 'silicon')
        
        print(f"Data validator run initialized:")
        print(f"  Config: {self.config_file}")
        print(f"  Module type: {module_type}")
        print(f"  Input CSV: {self.input_csv}")
        print(f"  Clean data directory: {self.clean_data_dir}")
        print(f"  Validation results directory: {self.validation_results_dir}")
    
    def load_config(self) -> dict:
        """Load test configuration from JSON file"""
        # Check if config_file is a full path or just filename
        if Path(self.config_file).is_absolute() or '/' in self.config_file or '\\' in self.config_file:
            # Full path provided
            config_path = Path(self.config_file)
        else:
            # Just filename provided, look in results/model_configs/
            config_path = Path(__file__).parent.parent / "results" / "model_configs" / self.config_file
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def setup_paths(self):
        """Setup all necessary paths based on configuration"""
        # Get date range from config
        date_sel = self.config.get('date_selection', {})
        start = date_sel.get('start', 'unknownStart').replace('-', '')
        end = date_sel.get('end', 'unknownEnd').replace('-', '')
        config_name = self.config.get('model_name', 'unknownConfig')
        
        # Get module_type from config
        module_type = self.config.get('module_type', 'silicon')
        
        # Setup directories based on module_type
        raw_data_dir = Path(__file__).parent.parent / "results" / "training_data" / "rawData" / f"{start}_{end}"
        
        # Use module-type specific clean data directory
        if module_type in ['silicon', 'perovskite']:
            self.clean_data_dir = Path(__file__).parent.parent / "results" / "training_data" / module_type.capitalize() / "cleanData" / f"{start}_{end}"
        else:
            # Fallback to general directory for unknown module types
            self.clean_data_dir = Path(__file__).parent.parent / "results" / "training_data" / "cleanData" / f"{start}_{end}"
        
        self.validation_results_dir = self.clean_data_dir / "validation_results"
        
        # Create directories
        self.clean_data_dir.mkdir(parents=True, exist_ok=True)
        self.validation_results_dir.mkdir(parents=True, exist_ok=True)
        
        # Find input CSV file
        self.input_csv = raw_data_dir / f"{start}_{end}_{config_name}_raw.csv"
        
        # Setup output paths
        self.clean_data_path = self.clean_data_dir / f"{start}_{end}_{config_name}_clean.csv"
        self.report_path = self.validation_results_dir / f"{start}_{end}_{config_name}_validation_report.txt"
        
        # Extract features from config
        self.features = self.config.get('features', [])
        self.outputs = self.config.get('output', [])
        self.time_features = self.config.get('time_features', ["day_of_year", "month", "weekday", "hour", "minute"])
        self.dataset_name = self.input_csv.stem
    
    def test_data_loading(self) -> Optional[pd.DataFrame]:
        """Test loading data from CSV file"""
        print("Loading training data...")
        
        try:
            if not self.input_csv.exists():
                raise FileNotFoundError(f"Input CSV file not found: {self.input_csv}")
            
            df = pd.read_csv(self.input_csv)
            print(f"OK CSV loaded successfully: {len(df)} rows, {len(df.columns)} columns")
            
            # Check for time column
            if "_time" in df.columns:
                # Optimized timestamp conversion with explicit UTC handling
                df["_time"] = pd.to_datetime(df["_time"], utc=True, errors='coerce')
                print("OK '_time' column found and converted to datetime")
            else:
                print("FAIL Missing column '_time' in CSV!")
                return None
            
            return df
            
        except Exception as e:
            print(f"FAIL Failed to load CSV: {e}")
            return None
    
    def test_validator_initialization(self) -> Optional[DataValidator]:
        """Test DataValidator initialization"""
        print("Initializing DataValidator...")
        
        try:
            # Get module_type from config
            module_type = self.config.get('module_type', 'silicon')
            print(f"Using module_type from config: {module_type}")
            
            # Initialize DataValidator in standard mode for the configured module type
            # (Standard mode corresponds to the old avg_mode=True behavior)
            validator = DataValidator(
                output_dir=str(self.validation_results_dir),
                extended_avg_mode=True,  # Standard mode (simple average calculation)
                module_type_validation=module_type,
                raw_data_path=str(self.input_csv)
            )
            print(f"OK DataValidator initialized (Standard mode for {module_type})")
            return validator
            
        except Exception as e:
            print(f"FAIL Failed to initialize DataValidator: {e}")
            return None
    
    def test_data_validation(self, validator: DataValidator, df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[dict]]:
        """Test data validation process"""
        print("Running data validation...")
        
        try:
            cleaned_data, summary = validator.validate_training_data(
                data=None,  # In standard mode, the DataFrame is ignored
                dataset_name=self.dataset_name,
                features=self.features,
                outputs=self.outputs,
                time_features=self.time_features
            )
            print("OK Validation completed (Standard mode)")
            return cleaned_data, summary
            
        except Exception as e:
            print(f"FAIL Validation failed: {e}")
            return None, None
    
    def save_results(self, cleaned_data: pd.DataFrame, validator: DataValidator, original_df: pd.DataFrame):
        """Save validation results"""
        print("Saving validation results...")

        # Save cleaned data and create resolution variants
        try:
            # Save original 5min version
            cleaned_data.to_csv(self.clean_data_path, index=False)
            print(f"OK Cleaned data saved: {self.clean_data_path}")
            
            # Create and save resolution variants
            self._create_resolution_variants(cleaned_data)
            
        except Exception as e:
            print(f"FAIL Failed to save cleaned data: {e}")

        # Save validation report
        try:
            with open(self.report_path, "w") as f:
                f.write(validator.get_validation_report())
            print(f"OK Validation report saved: {self.report_path}")
        except Exception as e:
            print(f"FAIL Failed to save report: {e}")

        print(f"OK Cleaned data: {len(cleaned_data)} of {len(original_df)} rows remaining.")
    
    def _create_resolution_variants(self, data_5min: pd.DataFrame):
        """
        Create and save 5min, 10min, and 1h resolution variants of the cleaned data.
        
        Args:
            data_5min (pd.DataFrame): Original 5-minute resolution data
        """
        print("Creating resolution variants...")
        
        # Debug: Check if P_normalized is in the data
        if 'P_normalized' in data_5min.columns:
            print("OK P_normalized column found in data for resolution variants")
        else:
            print("WARNING P_normalized column NOT found in data for resolution variants")
            print(f"Available columns: {list(data_5min.columns)}")
        
        # Get base filename without extension
        base_path = str(self.clean_data_path).replace('.csv', '')
        
        # Apply correct column ordering for 5min data
        id_columns = ["_time", "module_type"]
        ordered_time_features = ["day_of_year", "month", "weekday", "hour", "minute"]
        ordered_features = ["P", "P_normalized", "Irr"]
        # Add cross-module-type Irr columns
        cross_irr_features = ["Irr_si", "Irr_pvk"]
        
        final_columns = []
        for col in id_columns + ordered_time_features + ordered_features + cross_irr_features:
            if col in data_5min.columns and col not in final_columns:
                final_columns.append(col)
        
        data_5min_ordered = data_5min[final_columns]
        
        # Save 5min version (rename original file)
        path_5min = base_path + '-5min.csv'
        data_5min_ordered.to_csv(path_5min, index=False)
        print(f"OK 5min resolution saved: {path_5min}")
        
        # Create 10min version (aggregate every 2 records)
        data_10min = self._resample_pv_data(data_5min, "10min")
        path_10min = base_path + '-10min.csv'
        data_10min.to_csv(path_10min, index=False)
        print(f"OK 10min resolution saved: {path_10min}")
        
        # Create 1h version (aggregate every 12 records)
        data_1h = self._resample_pv_data(data_5min, "1h")
        path_1h = base_path + '-1h.csv'
        data_1h.to_csv(path_1h, index=False)
        print(f"OK 1h resolution saved: {path_1h}")
    
    def _resample_pv_data(self, data: pd.DataFrame, target_resolution: str) -> pd.DataFrame:
        """
        Resample PV data to target resolution using timestamp-based bins so that
        windows are exactly aligned (10min: HH:00/10/20/.., 1h: HH:00).
        """
        if data.empty:
            return data.copy()

        # Accept either 'timestamp' or '_time' from the 5min clean file
        original_ts_col = None
        if 'timestamp' in data.columns:
            original_ts_col = 'timestamp'
        elif '_time' in data.columns:
            original_ts_col = '_time'
        else:
            raise ValueError("Expected 'timestamp' or '_time' column in cleaned data for time-based resampling")

        df = data.copy()
        # Normalize to 'timestamp' for processing
        if original_ts_col == '_time':
            df['timestamp'] = pd.to_datetime(df['_time'])
        else:
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        if target_resolution == '10min':
            bin_ts = df['timestamp'].dt.floor('10min')
        elif target_resolution == '1h':
            bin_ts = df['timestamp'].dt.ceil('H')
        else:
            raise ValueError(f"Unsupported target resolution: {target_resolution}")

        df['_bin'] = bin_ts

        # DEBUG: Show which timestamps are grouped together for resampling
        if target_resolution == '1h':
            print(f"\n=== DEBUG: 1h Resampling ===")
            print(f"Original data: {len(df)} records")
            
            # Show first 3 bins as examples
            unique_bins = sorted(df['_bin'].unique())
            print(f"Total bins: {len(unique_bins)}")
            
            for i, bin_ts in enumerate(unique_bins[:5]):  # Show first 3 bins
                bin_data = df[df['_bin'] == bin_ts]
                original_timestamps = bin_data['timestamp'].tolist()
                print(f"Bin {i+1} ({bin_ts}): {len(bin_data)} records")
                print(f"  Original timestamps: {original_timestamps}")
                print(f"  Resulting timestamp: {bin_ts}")
                print()

        # Define feature groups
        time_features = ["day_of_year", "month", "weekday", "hour", "minute"]
        pv_features = ["Temp", "U", "I", "P", "P_normalized", "AmbTemp", "AmbHmd", "Irr"]
        # Preserve all other columns (e.g., module_type) using first value per bin
        other_cols = [c for c in df.columns if c not in pv_features + time_features + ['timestamp', '_bin']]

        agg_dict = {}
        for col in pv_features:
            if col in df.columns:
                agg_dict[col] = 'mean'
        for col in time_features:
            if col in df.columns:
                agg_dict[col] = 'first'
        for col in other_cols:
            agg_dict[col] = 'first'
        # Quantile scaling is now applied in the Data Validator itself
        
        # Always keep timestamp-bin
        resampled = df.groupby('_bin').agg(agg_dict).reset_index().rename(columns={'_bin': 'timestamp'})

        # Derive time features correctly from the bin (and leave preserved columns unchanged)
        resampled['day_of_year'] = resampled['timestamp'].dt.dayofyear
        resampled['month'] = resampled['timestamp'].dt.month
        resampled['weekday'] = resampled['timestamp'].dt.weekday
        resampled['hour'] = resampled['timestamp'].dt.hour
        resampled['minute'] = resampled['timestamp'].dt.minute

        # For 1h: Set minute to 0
        if target_resolution == '1h' and 'minute' in resampled.columns:
            resampled['minute'] = 0

        # Rename 'timestamp' back to original column name
        if original_ts_col == '_time':
            resampled['_time'] = resampled['timestamp']
        # Column order: like original (5min), only resolution adjusted
        original_cols = list(data.columns)
        # Replace ts col name in original order with current one
        ordered_cols = []
        for col in original_cols:
            if col == '_time' and original_ts_col == '_time':
                ordered_cols.append('_time')
            elif col == 'timestamp' and original_ts_col == 'timestamp':
                ordered_cols.append('timestamp')
            elif col != '_time' and col != 'timestamp':
                if col in resampled.columns:
                    ordered_cols.append(col)
        # Prepend correct ts column if not already first
        ts_name = original_ts_col
        if ordered_cols and ordered_cols[0] != ts_name:
            if ts_name in ordered_cols:
                ordered_cols.remove(ts_name)
            ordered_cols = [ts_name] + ordered_cols
        # Set correct column order: _time, module_type, time_features, P, P_normalized, Irr, cross-module Irr
        id_columns = ["_time", "module_type"]
        ordered_time_features = ["day_of_year", "month", "weekday", "hour", "minute"]
        ordered_features = ["P", "P_normalized", "Irr"]
        # Add cross-module-type Irr columns
        cross_irr_features = ["Irr_si", "Irr_pvk"]
        
        final_columns = []
        for col in id_columns + ordered_time_features + ordered_features + cross_irr_features:
            if col in resampled.columns and col not in final_columns:
                final_columns.append(col)
        
        resampled = resampled[final_columns]

        print(f"Resampled from {len(data)} to {len(resampled)} records ({target_resolution})")
        return resampled
    
    def run_full_validation(self) -> bool:
        """Run the complete data validator validation"""
        print("="*50)
        print("DATA VALIDATOR RUN (NEW MODULAR STRUCTURE)")
        print("="*50)
        print(f"Configuration: {self.config_file}")
        print(f"Model: {self.config.get('model_name', 'unknown')}")
        print(f"Module type: {self.config.get('module_type', 'silicon')}")
        
        # Test 1: Data loading
        df = self.test_data_loading()
        if df is None:
            return False
        
        # Test 2: Validator initialization
        validator = self.test_validator_initialization()
        if validator is None:
            return False
        
        # Test 3: Data validation
        cleaned_data, summary = self.test_data_validation(validator, df)
        if cleaned_data is None:
            return False
        
        # Test 4: Save results
        self.save_results(cleaned_data, validator, df)
        
        print("\n" + "="*50)
        print("OK ALL VALIDATION STEPS COMPLETED")
        print("="*50)
        return True


def main():
    """Main function to run the data validator"""
    # Check command line arguments
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = "test_lstm_model_settings.json"
    
    validator_run = RunDataValidatorNew(config_file)
    
    try:
        success = validator_run.run_full_validation()
        if success:
            print("SUCCESS: Data validator run completed successfully")
            sys.exit(0)
        else:
            print("ERROR: Data validator run failed")
            sys.exit(1)
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
