"""
Standard validation mode - simple averaging with basic validation.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

from ..validation_methods.range_validation import RangeValidator
from ..utils.validation_rules import ValidationRules
from ..quantile_scaling.scaling_calculator import QuantileScalingCalculator
from ..irr_gap_filling.irr_gap_filler import IrrGapFiller
from ..report_generation.report_integration import create_step_tracker

logger = logging.getLogger('influxData_api.data_validator_new')


class StandardMode:
    """
    Standard validation mode that performs simple averaging with basic validation.
    
    This mode:
    - Loads raw data
    - Performs basic range validation
    - Creates averages from individual modules
    - Skips outlier detection and module failure detection
    """
    
    def __init__(self, file_manager, data_processor, module_type_validation: str, 
                 raw_data_path: str, validation_mode: str = "training"):
        """
        Initialize StandardMode.
        
        Args:
            file_manager: FileManager instance
            data_processor: DataProcessor instance
            module_type_validation: Module type ('silicon' or 'perovskite')
            raw_data_path: Path to raw data CSV
            validation_mode: Validation mode ('training' or 'forecast')
        """
        self.file_manager = file_manager
        self.data_processor = data_processor
        self.module_type_validation = module_type_validation
        self.raw_data_path = raw_data_path
        self.validation_mode = validation_mode
        self.quantile_scaler = QuantileScalingCalculator()
        self.irr_gap_filler = IrrGapFiller()
        
        # Initialize validation results
        self.validation_results = {
            "range_violations": [],
            "outliers": [],
            "module_failures": [],
            "summary": {}
        }
        
        # Initialize range validator
        self.range_validator = RangeValidator(ValidationRules.get_value_ranges(module_type_validation))
    
    def validate(self, data: Optional[Any], dataset_name: str, features: List[str],
                outputs: Optional[List[str]] = None, time_features: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Perform standard validation with simple averaging.
        
        Args:
            data: Input data (ignored for avg operations)
            dataset_name: Name of the dataset
            features: List of feature columns
            outputs: List of output columns
            time_features: List of time features
            
        Returns:
            Tuple of cleaned data and validation summary
        """
        logger.info("STANDARD MODE aktiv: Es werden nur Range-Validation und Duplicate-Aggregation durchgeführt. Outlier- und Failure-Detection werden übersprungen!")
        
        # Initialize report tracker
        output_dir = Path(self.raw_data_path).parent / "validation_results"
        report_tracker = create_step_tracker(dataset_name, output_dir)
        
        # Load raw data
        filtered_data = self._load_and_prepare_raw_data(dataset_name)
        initial_record_count = len(filtered_data)
        
        # FIRST: Add time features before validation (needed for Irr filling)
        logger.info("Adding time features before validation...")
        time_features = ['day_of_year', 'month', 'weekday', 'hour', 'minute']
        filtered_data = self.data_processor.add_time_features(filtered_data, time_features)
        
        # SECOND: Apply validation to individual modules
        logger.info("Applying range validation to individual modules...")
        range_violations = self.range_validator.validate_value_ranges(filtered_data, ['P', 'Irr'])
        # Remove time stamps with range violations instead of interpolating
        filtered_data_after_range = self.range_validator.remove_violation_timestamps(filtered_data, range_violations)
        
        # Track range validation step
        report_tracker.track_range_validation(
            input_records=len(filtered_data),
            output_records=len(filtered_data_after_range),
            removed_records=len(filtered_data) - len(filtered_data_after_range),
            violations=range_violations,
            data_sample=filtered_data_after_range.head(1000) if len(filtered_data_after_range) > 0 else None
        )
        filtered_data = filtered_data_after_range
        
        # FOURTH: Calculate scaling factors from validated individual modules
        logger.info("Calculating scaling factors from validated individual modules...")
        scaling_factors = self.quantile_scaler.calculate_scaling_factors(filtered_data)
        
        # FIFTH: Apply quantile scaling to individual modules
        logger.info("Applying quantile scaling to individual modules...")
        filtered_data_scaled = self.quantile_scaler.apply_scaling(filtered_data, scaling_factors)
        
        # Track quantile scaling step
        p_column_generated = 'P_normalized' in filtered_data_scaled.columns
        report_tracker.track_quantile_scaling(
            input_records=len(filtered_data),
            output_records=len(filtered_data_scaled),
            scaling_factors=scaling_factors,
            p_normalized_generated=p_column_generated,
            data_sample=filtered_data_scaled.head(1000) if len(filtered_data_scaled) > 0 else None
        )
        
        # SIXTH: Create average data from scaled individual modules
        numeric_columns = ['Temp', 'U', 'I', 'P', 'P_normalized', 'AmbTemp', 'AmbHmd', 'Irr']
        available_numeric_columns = [col for col in numeric_columns if col in filtered_data_scaled.columns]
        
        # DEBUG: Check P values before averaging
        logger.info(f"DEBUG: P-Werte vor Averaging: {filtered_data_scaled['P'].describe()}")
        logger.info(f"DEBUG: Anzahl Module vor Averaging: {len(filtered_data_scaled['Name'].unique()) if 'Name' in filtered_data_scaled.columns else 'No Name column'}")
        logger.info(f"DEBUG: Beispiel P-Werte erster Timestamp: {filtered_data_scaled[filtered_data_scaled['_time'] == filtered_data_scaled['_time'].iloc[0]]['P'].tolist()}")
        
        # DEBUG: Show timestamp and module details for averaging (only first 2 timestamps)
        unique_timestamps = sorted(filtered_data_scaled['_time'].unique())
        logger.info(f"DEBUG: Alle Zeitstempel für Averaging ({len(unique_timestamps)} total): {unique_timestamps[:10]}...")
        
        for i, timestamp in enumerate(unique_timestamps[:2]):  # Only show first 2 timestamps
            timestamp_data = filtered_data_scaled[filtered_data_scaled['_time'] == timestamp]
            module_names = timestamp_data['Name'].unique().tolist() if 'Name' in timestamp_data.columns else ['No Name column']
            p_values = timestamp_data['P'].tolist()
            logger.info(f"DEBUG: Zeitstempel {i+1} ({timestamp}): Module={module_names}, P-Werte={p_values}")
        
        avg_data = self._calculate_module_type_averages_with_scaling(filtered_data_scaled, available_numeric_columns, self.module_type_validation)
        avg_data["module_type"] = self.module_type_validation
        
        # Track aggregation step
        aggregation_stats = {
            'aggregation_method': 'Robust Average (with outlier exclusion)',
            'data_reduction_ratio': f'{len(filtered_data_scaled)/len(avg_data):.1f}:1' if len(avg_data) > 0 else 'N/A',
            'temporal_resolution': '5-minute intervals'
        }
        
        step_data = {
            'description': 'Data aggregation (Robust Average with outlier exclusion)',
            'input_records': len(filtered_data_scaled),
            'output_records': len(avg_data),
            'removed_records': len(filtered_data_scaled) - len(avg_data),
            'statistics': aggregation_stats,
            'data_sample': avg_data.head(1000) if len(avg_data) > 0 else None
        }
        report_tracker.reporter.add_validation_step("Creating Averages from Validated Individual Module Data", step_data)
        
        # Add cross-module Irr values (Irr_si or Irr_pvk)
        avg_data_before_cross = avg_data.copy()
        avg_data = self._add_cross_module_irr_values(avg_data, self.module_type_validation)
        
        # Track cross-module Irr addition
        irr_si_added = len(avg_data[avg_data['Irr_si'].notna()]) if 'Irr_si' in avg_data.columns else 0
        irr_pvk_added = len(avg_data[avg_data['Irr_pvk'].notna()]) if 'Irr_pvk' in avg_data.columns else 0
        
        report_tracker.track_cross_module_irr(
            input_records=len(avg_data_before_cross),
            output_records=len(avg_data),
            irr_si_added=irr_si_added,
            irr_pvk_added=irr_pvk_added,
            data_sample=avg_data.head(1000) if len(avg_data) > 0 else None
        )

        # Add time features and select columns (ensure all time features are present)
        time_features_to_add = ['day_of_year', 'month', 'weekday', 'hour', 'minute']
        avg_data = self.data_processor.add_time_features(avg_data, time_features_to_add)

        # THIRD: Fill missing Irr values
        avg_data_before_gap = avg_data.copy()
        missing_irr_before = avg_data['Irr'].isna().sum()
        avg_data = self.irr_gap_filler.fill_missing_irr_with_gs10(avg_data)
        missing_irr_after = avg_data['Irr'].isna().sum()
        filled_values = missing_irr_before - missing_irr_after
        
        # Track gap filling step
        report_tracker.track_irr_gap_filling(
            input_records=len(avg_data_before_gap),
            output_records=len(avg_data),
            missing_values=missing_irr_before,
            filled_values=filled_values,
            data_sample=avg_data.head(1000) if len(avg_data) > 0 else None
        )
        
        # FOURTH: FINAL Correlation validation after gap filling
        logger.info("Step 4: Final correlation validation after gap filling...")
        from ..validation_methods import CorrelationValidator
        from ..utils.validation_rules import ValidationRules
        correlation_validator = CorrelationValidator(ValidationRules.CORRELATION_CONFIG)
        
        avg_data_before_final_corr = avg_data.copy()
        final_correlation_anomalies = correlation_validator.detect_correlation_anomalies(avg_data, f"{dataset_name}_final")
        if len(final_correlation_anomalies) > 0:
            logger.info(f"Detected {len(final_correlation_anomalies)} correlation anomalies after gap filling")
            logger.info("DEBUG: First 10 post-gap-filling correlation anomalies:")
            for i, anomaly in enumerate(final_correlation_anomalies[:10]):
                logger.info(f"  Final Anomaly {i+1}: {anomaly['timestamp']}")
                logger.info(f"    Irr change: {anomaly['irr_change_pct']:.1f}%, P change: {anomaly['p_change_pct']:.1f}%")
                logger.info(f"    Proportion: {anomaly['proportion']:.1f}% (min required: {anomaly['min_proportion_pct']:.1f}%)")
                logger.info(f"    Reason: {anomaly['description']}")
            
            # Remove final correlation anomalies
            avg_data = correlation_validator.remove_correlation_anomalies(avg_data, final_correlation_anomalies)
            logger.info(f"Removed {len(final_correlation_anomalies)} final correlation anomaly records")
        else:
            logger.info("No correlation anomalies detected after gap filling")
        
        # Track final correlation validation
        report_tracker.track_correlation_validation_new(
            input_records=len(avg_data_before_final_corr),
            output_records=len(avg_data),
            removed_records=len(avg_data_before_final_corr) - len(avg_data),
            anomalies_detected=len(final_correlation_anomalies),
            data_sample=avg_data.head(1000) if len(avg_data) > 0 else None,
            step_name="Final Correlation Validation (Post Gap-Filling)"
        )
        
        # Save scaling factors
        logger.info(f"About to save scaling factors: {scaling_factors}")
        self.quantile_scaler.save_scaling_factors(scaling_factors, dataset_name, self.module_type_validation, self.file_manager.output_dir)
        
        # Step 5: Generate summary and save results
        logger.info("Step 5: Generating validation summary...")
        summary = self._generate_validation_summary(filtered_data, avg_data, dataset_name)
        self.validation_results["summary"] = summary
        
        # Add summary statistics to report
        summary_stats = {
            'total_initial_records': initial_record_count,
            'total_final_records': len(avg_data),
            'total_removed_records': initial_record_count - len(avg_data),
            'data_retention_rate': (len(avg_data) / initial_record_count) * 100 if initial_record_count > 0 else 0,
            'validation_mode': 'Standard Mode'
        }
        report_tracker.add_summary_statistics(summary_stats)
        
        # Generate report
        try:
            report_path = report_tracker.generate_report()
            logger.info(f"Validation report generated: {report_path}")
        except Exception as e:
            logger.warning(f"Failed to generate validation report: {e}")
        
        # Save validation results
        self.file_manager.save_validation_results(dataset_name, self.validation_results)
        
        # Save final cleaned data
        self._save_final_cleaned_data(avg_data, dataset_name)
        
        # Select final columns in the correct order
        id_columns = ["_time", "module_type"]
        ordered_time_features = ["day_of_year", "month", "weekday", "hour", "minute"]
        ordered_features = ["P", "P_normalized", "Irr"]
        # Add cross-module-type Irr columns
        cross_irr_features = ["Irr_si", "Irr_pvk"]
        final_columns = []
        for col in id_columns + ordered_time_features + ordered_features + cross_irr_features:
            if col in avg_data.columns and col not in final_columns:
                final_columns.append(col)
        avg_data = avg_data[final_columns]
        
        # Save the average CSV with time features
        output_path = self.file_manager.get_output_path(dataset_name, self.module_type_validation, "standard")
        avg_data.to_csv(output_path, index=False)
        logger.info(f"Average-CSV mit Time-Features gespeichert: {output_path}")
        
        return avg_data, summary
    
    def _load_and_prepare_raw_data(self, dataset_name: str) -> pd.DataFrame:
        """
        Load and prepare raw data for standard mode processing.
        
        Returns:
            DataFrame with averaged data
        """
        raw_data_path = Path(self.raw_data_path)
        if not raw_data_path.exists():
            raise FileNotFoundError(f"Rohdaten-CSV nicht gefunden: {raw_data_path}")
        
        logger.info(f"Lade Rohdaten für Average-Berechnung: {raw_data_path}")
        raw_data = pd.read_csv(raw_data_path)
        
        if "_time" in raw_data.columns:
            # Zeitstempel-Konvertierung: Entferne Zeitzonen-Offset vor der Konvertierung
            if raw_data["_time"].dtype == 'object':
                # Entferne Zeitzonen-Offset mit Regex (z.B. "+01:00", "+02:00", "Z")
                raw_data["_time"] = raw_data["_time"].str.replace(r'[+-]\d{2}:\d{2}$|Z$', '', regex=True)
            
            # Jetzt konvertiere zu datetime (ohne Zeitzonen-Probleme)
            raw_data["_time"] = pd.to_datetime(raw_data["_time"], errors='coerce')
        
        # Unify module names und setze module_type
        raw_data = self.data_processor.unify_module_names(raw_data)
        if "Name" in raw_data.columns:
            raw_data["module_type"] = raw_data["Name"].apply(self.data_processor.get_module_type_from_name)
        
        # Remove Perovskite exclusion periods if applicable
        if self.module_type_validation == "perovskite":
            raw_data = self.data_processor.remove_perovskite_exclusion_periods(raw_data)
        
        # Filter only modules of the desired type
        filtered_data = raw_data[raw_data['module_type'] == self.module_type_validation].copy()
        
        if filtered_data.empty:
            raise ValueError(f"Keine {self.module_type_validation}-Module in den Rohdaten gefunden!")
        
        logger.info(f"Erstelle Average-CSV für {self.module_type_validation}-Module mit einfacher Durchschnittsberechnung...")
        
        return filtered_data
    
    
    
    
    def _save_final_cleaned_data(self, data: pd.DataFrame, dataset_name: str):
        """
        Save final cleaned data to a CSV file.
        
        Args:
            data: Cleaned data
            dataset_name: Dataset name for file naming
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cleaned_path = self.file_manager.output_dir / f"{dataset_name}_05_FINAL_CLEANED_DATA_{timestamp}.csv"
        data.to_csv(cleaned_path, index=False)
        
        logger.info(f"Final cleaned data saved to: {cleaned_path}")
    
    def _calculate_module_type_averages_with_scaling(self, module_data: pd.DataFrame, numeric_columns: List[str], module_type: str) -> pd.DataFrame:
        """
        Calculate averages for a specific module type across all timestamps.
        The input data should already have P_normalized calculated from quantile scaling.
        
        Args:
            module_data (pd.DataFrame): Data for one module type (already scaled with P_normalized)
            numeric_columns (List[str]): Columns to average (including P_normalized)
            module_type (str): Type of module ('silicon' or 'perovskite')
            
        Returns:
            pd.DataFrame: Average values per timestamp including P_normalized
        """
        # Ensure we only aggregate numeric columns that exist in the data
        available_numeric_columns = [col for col in numeric_columns if col in module_data.columns]
        
        if not available_numeric_columns:
            logger.warning(f"No numeric columns available for {module_type} modules")
            # Return empty DataFrame with required structure
            return pd.DataFrame(columns=['_time', 'module_type'] + available_numeric_columns)
        
        # Select only the columns we want to aggregate
        columns_to_aggregate = ['_time', 'Name'] + available_numeric_columns
        aggregation_data = module_data[columns_to_aggregate].copy()
        
        # OPTIMIZATION: Use pandas groupby operations instead of iterating over timestamps
        logger.info(f"Starting optimized average calculation for {module_type} modules with {len(aggregation_data['_time'].unique())} timestamps")
        
        # Group by timestamp and apply robust aggregation
        grouped = aggregation_data.groupby('_time')
        averages_list = []
        total_exclusions = 0
        
        # Process in chunks to avoid memory issues
        chunk_size = 1000
        timestamp_chunks = [list(aggregation_data['_time'].unique())[i:i+chunk_size] 
                          for i in range(0, len(aggregation_data['_time'].unique()), chunk_size)]
        
        for chunk_idx, timestamp_chunk in enumerate(timestamp_chunks):
            logger.info(f"Processing chunk {chunk_idx + 1}/{len(timestamp_chunks)} ({len(timestamp_chunk)} timestamps)")
            
            for timestamp in timestamp_chunk:
                timestamp_data = aggregation_data[aggregation_data['_time'] == timestamp]
                
                # Initialize row for this timestamp
                avg_row = {'_time': timestamp, 'module_type': module_type}
                
                for column in available_numeric_columns:
                    # Special handling for Irr column - use only specific reference modules
                    if column == 'Irr':
                        # Define reference modules for Irr values
                        if module_type == 'silicon':
                            reference_modules = ['Sanyo_5-1', 'Sanyo_5_1']
                        elif module_type == 'perovskite':
                            reference_modules = ['Perovskite_1_1', 'Perovskite_1']  # Include older naming variant
                        else:
                            reference_modules = []
                        
                        # Filter timestamp_data to only include reference modules for Irr
                        if reference_modules:
                            irr_data = timestamp_data[timestamp_data['Name'].isin(reference_modules)]
                            if len(irr_data) > 0:
                                values = irr_data[column].dropna()
                                logger.debug(f"Using Irr from reference modules {reference_modules}: {len(values)} values")
                            else:
                                # logger.warning(f"No reference modules {reference_modules} found for Irr at timestamp {timestamp}")
                                values = pd.Series(dtype=float)  # Empty series
                        else:
                            logger.warning(f"No reference modules defined for module_type {module_type}")
                            values = pd.Series(dtype=float)  # Empty series
                        
                        original_count = len(timestamp_data[column])
                        null_count = original_count - len(values)
                    else:
                        # Normal averaging for all other columns (P, Temp, etc.)
                        values = timestamp_data[column].dropna()
                        original_count = len(timestamp_data[column])
                        null_count = original_count - len(values)
                    
                    if len(values) == 0:
                        # No valid values for this timestamp and column
                        avg_row[column] = None
                        total_exclusions += null_count
                        continue
                    
                    # Check for suspicious zero values when other modules have non-zero values
                    if len(values) > 1:
                        non_zero_values = values[values != 0.0]
                        zero_values = values[values == 0.0]
                        
                        # If we have both zero and non-zero values, exclude zeros as suspicious
                        if len(non_zero_values) > 0 and len(zero_values) > 0:
                            values = non_zero_values
                            total_exclusions += len(zero_values)
                    
                    if len(values) == 0:
                        # All values were excluded as suspicious zeros
                        avg_row[column] = None
                        total_exclusions += null_count
                        continue
                    
                    if len(values) == 1:
                        # Only one module has valid data, use that value
                        avg_row[column] = values.iloc[0]
                        total_exclusions += null_count
                        continue
                    
                    # Multiple modules have data - check for outliers
                    if len(values) >= 3:  # Need at least 3 values for outlier detection
                        # Calculate Z-scores for outlier detection
                        mean_val = values.mean()
                        std_val = values.std()
                        
                        if std_val > 0:  # Avoid division by zero
                            z_scores = abs((values - mean_val) / std_val)
                            
                            # Remove outliers (Z-score > 2.5 is considered outlier)
                            outlier_threshold = 2.5
                            non_outlier_mask = z_scores <= outlier_threshold
                            clean_values = values[non_outlier_mask]
                            outlier_count = len(values) - len(clean_values)
                            
                            if len(clean_values) > 0:
                                # Use median for robustness against remaining outliers
                                avg_row[column] = clean_values.median()
                            else:
                                # All values were outliers, use median of original values
                                avg_row[column] = values.median()
                        else:
                            # No variation in values, use mean
                            avg_row[column] = values.mean()
                    else:
                        # Not enough values for outlier detection, use median
                        avg_row[column] = values.median()
                
                # ADDITIONAL: Add cross-module-type Irr values (Irr_si or Irr_pvk)
                # Get Irr from the OTHER module type
                if module_type == 'silicon':
                    # Add Irr_pvk (from Perovskite_1_1 or Perovskite_1 for older data)
                    other_modules = ['Perovskite_1_1', 'Perovskite_1']
                    other_column_name = 'Irr_pvk'
                elif module_type == 'perovskite':
                    # Add Irr_si (from Sanyo_5-1 or Sanyo_5_1)
                    other_modules = ['Sanyo_5-1', 'Sanyo_5_1']
                    other_column_name = 'Irr_si'
                else:
                    other_modules = []
                    other_column_name = None
                
                if other_modules and other_column_name and 'Irr' in timestamp_data.columns:
                    other_irr_data = timestamp_data[timestamp_data['Name'].isin(other_modules)]
                    if len(other_irr_data) > 0:
                        other_values = other_irr_data['Irr'].dropna()
                        if len(other_values) > 0:
                            avg_row[other_column_name] = other_values.median()
                            logger.debug(f"Added {other_column_name} from modules {other_modules}: {other_values.median()}")
                        else:
                            avg_row[other_column_name] = None
                    else:
                        avg_row[other_column_name] = None
                        logger.debug(f"No {other_modules} modules found for {other_column_name} at timestamp {timestamp}")
                
                averages_list.append(avg_row)
        
        # Convert to DataFrame
        averages_df = pd.DataFrame(averages_list)
        
        # Sort by timestamp
        averages_df = averages_df.sort_values('_time').reset_index(drop=True)
        
        # P_normalized should already be included in the averages if it was in the input data
        # The robust averaging logic above already handles all columns in available_numeric_columns
        if 'P_normalized' in available_numeric_columns:
            logger.info(f"P_normalized column included in averages for {module_type} modules")
        else:
            logger.warning(f"P_normalized not found in available columns: {available_numeric_columns}")
        
        logger.info(f"Robust average calculation for {module_type} modules: {total_exclusions} data points excluded (null values + suspicious zeros + outliers)")
        return averages_df
    
    def _add_cross_module_irr_values(self, avg_data: pd.DataFrame, module_type: str) -> pd.DataFrame:
        """
        Add cross-module Irr values (Irr_si or Irr_pvk) to the average data.
        
        Args:
            avg_data: DataFrame with averaged data for the current module type
            module_type: Current module type ('silicon' or 'perovskite')
            
        Returns:
            DataFrame with added cross-module Irr columns
        """
        # Determine which cross-module Irr to add
        if module_type == 'silicon':
            other_modules = ['Perovskite_1_1', 'Perovskite_1']  # Include older naming variant
            other_column_name = 'Irr_pvk'
        elif module_type == 'perovskite':
            other_modules = ['Sanyo_5-1', 'Sanyo_5_1']
            other_column_name = 'Irr_si'
        else:
            # No cross-module Irr for unknown module types
            return avg_data
        
        # Load raw data
        raw_data_path = Path(self.raw_data_path)
        if not raw_data_path.exists():
            logger.warning(f"Raw data not found for cross-module Irr: {raw_data_path}")
            avg_data[other_column_name] = None
            return avg_data
        
        try:
            raw_data = pd.read_csv(raw_data_path)
            if "_time" in raw_data.columns:
                # Use EXACTLY the same timestamp processing as CSV_Handler
                logger.info(f"DEBUG: Processing timestamps exactly like CSV_Handler")
                
                # Remove timezone offsets before conversion (same as CSV_Handler)
                if raw_data["_time"].dtype == 'object':
                    logger.info(f"DEBUG: Removing timezone strings from raw data timestamps")
                    raw_data["_time"] = raw_data["_time"].str.replace(r'[+-]\d{2}:\d{2}$|Z$', '', regex=True)
                
                # Convert to datetime (same as CSV_Handler)
                raw_data["_time"] = pd.to_datetime(raw_data["_time"], errors='coerce')
                logger.info(f"DEBUG: Timestamps processed using CSV_Handler method")
            else:
                logger.warning("No '_time' column in raw data for cross-module Irr")
                avg_data[other_column_name] = None
                return avg_data
        except Exception as e:
            logger.warning(f"Error loading raw data for cross-module Irr: {e}")
            avg_data[other_column_name] = None
            return avg_data
        
        # Filter raw data for the other module types
        other_module_data = raw_data[raw_data['Name'].isin(other_modules)].copy()
        
        if len(other_module_data) == 0:
            logger.info(f"No {other_modules} modules found in raw data")
            avg_data[other_column_name] = None
            return avg_data
        
        # Calculate Irr values for each timestamp in avg_data
        cross_irr_values = []
        for _, row in avg_data.iterrows():
            timestamp = row['_time']
            
            # Find Irr values for this timestamp from other modules
            timestamp_data = other_module_data[other_module_data['_time'] == timestamp]
            
            if len(timestamp_data) > 0 and 'Irr' in timestamp_data.columns:
                irr_values = timestamp_data['Irr'].dropna()
                if len(irr_values) > 0:
                    cross_irr_value = irr_values.median()
                    logger.debug(f"Added {other_column_name} for {timestamp}: {cross_irr_value} from {len(irr_values)} modules")
                else:
                    cross_irr_value = None
            else:
                cross_irr_value = None
            
            cross_irr_values.append(cross_irr_value)
        
        # Add the cross-module Irr column to avg_data
        avg_data[other_column_name] = cross_irr_values
        
        logger.info(f"Added {other_column_name} column with {sum(1 for v in cross_irr_values if v is not None)} non-null values")
        return avg_data
    
    def _generate_validation_summary(self, original_data: pd.DataFrame, cleaned_data: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """
        Generate validation summary.
        
        Args:
            original_data: Original data before cleaning
            cleaned_data: Data after cleaning
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary with validation summary
        """
        summary = {
            "dataset_name": dataset_name,
            "original_records": len(original_data),
            "cleaned_records": len(cleaned_data),
            "records_removed": len(original_data) - len(cleaned_data),
            "removal_percentage": (len(original_data) - len(cleaned_data)) / len(original_data) * 100 if len(original_data) > 0 else 0,
            "module_type": self.module_type_validation,
            "validation_mode": "standard"
        }
        
        return summary
    
    def _save_final_cleaned_data(self, data: pd.DataFrame, dataset_name: str):
        """
        Save final cleaned data.
        
        Args:
            data: Cleaned data
            dataset_name: Name of the dataset
        """
        # This would save the data to the main training directory
        # Implementation depends on specific requirements
        pass
    
    def _select_final_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Select and order final columns.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with selected columns
        """
        id_columns = ["_time", "MAC", "Name", "module_type"]
        ordered_time_features = ["day_of_year", "month", "weekday", "hour", "minute"]
        ordered_features = ["Temp", "U", "I", "P", "P_normalized", "AmbTemp", "AmbHmd", "Irr"]
        
        final_columns = []
        for col in id_columns + ordered_time_features + ordered_features:
            if col in data.columns and col not in final_columns:
                final_columns.append(col)
        
        return data[final_columns]
    
    def _save_scaling_factors(self, scaling_factors: Dict[str, float], dataset_name: str, module_type: str):
        """
        Save scaling factors to file.
        
        Args:
            scaling_factors: Dictionary of scaling factors
            dataset_name: Name of the dataset
            module_type: Module type
        """
        scaling_path = self.file_manager.get_scaling_factors_path(dataset_name, module_type)
        
        # Convert to DataFrame and save
        scaling_df = pd.DataFrame(list(scaling_factors.items()), columns=['Module', 'Scaling_Factor'])
        scaling_df.to_csv(scaling_path, index=False)
        
        logger.info(f"Scaling factors saved: {scaling_path}")
    
    def get_validation_report(self) -> str:
        """
        Generate a comprehensive validation report.
        
        Returns:
            String containing the validation report
        """
        report = f"""
DataValidator Validation Report - Standard Mode
===============================================

Module Type: {self.module_type_validation}
Validation Mode: {self.validation_mode}
Raw Data Path: {self.raw_data_path}

Validation Results:
------------------
Range Violations: {len(self.validation_results['range_violations'])}
Outliers: {len(self.validation_results['outliers'])}
Module Failures: {len(self.validation_results['module_failures'])}

Summary:
--------
{self.validation_results['summary']}
"""
        return report
