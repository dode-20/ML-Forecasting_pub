"""
Correlation-based validation for PV data.

This module detects anomalies in the relationship between irradiance (Irr) and power (P)
by identifying timestamps where one parameter changes significantly while the other doesn't.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger('influxData_api.data_validator_new')


class CorrelationValidator:
    """
    Validates PV data by checking the correlation between irradiance and power.
    
    Detects anomalies where:
    - Irr changes significantly but P doesn't (or vice versa)
    - Sudden jumps in one parameter without corresponding change in the other
    - Unrealistic Irr-P relationships
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize CorrelationValidator.
        
        Args:
            config: Configuration dictionary for correlation validation
        """
        self.config = config
        self.min_proportion = config.get('min_proportion', 0.1)  # 10% mindest Proportion
        self.min_irr_threshold = config.get('min_irr_threshold', 100)  # 100 W/m² minimum Irr
    
    def detect_correlation_anomalies(self, data: pd.DataFrame, dataset_name: str) -> List[Dict[str, Any]]:
        """
        Detect correlation anomalies between Irr and P using simple pairwise comparison.
        
        Logic: If Irr changes by X%, P should also change by approximately X%.
        
        Args:
            data: Input DataFrame with Irr and P columns
            dataset_name: Name of the dataset
            
        Returns:
            List of detected correlation anomalies
        """
        anomalies = []
        
        if 'Irr' not in data.columns or 'P' not in data.columns:
            logger.warning("Irr or P columns not found for correlation validation")
            return anomalies
        
        if '_time' not in data.columns:
            logger.warning("_time column not found for correlation validation")
            return anomalies
        
        logger.info("Starting simplified correlation-based anomaly detection...")
        
        # Sort data by time to ensure proper temporal analysis
        data_sorted = data.sort_values('_time').reset_index(drop=True)
        
        # Filter data: only compare timestamps where both Irr and P are valid (> 0)
        # Note: We'll check the threshold later in pairs to allow transitions like 15→288
        valid_data = data_sorted[
            (data_sorted['Irr'] > 0) & 
            (data_sorted['P'] > 0)
        ].copy()
        
        if len(valid_data) < 2:
            logger.warning("Not enough valid data points for correlation validation")
            return anomalies
        
        logger.info(f"Comparing {len(valid_data)} valid data points for correlation anomalies (threshold {self.min_irr_threshold} W/m² applied per pair)...")
        
        # Compare each pair of consecutive timestamps
        for i in range(1, len(valid_data)):
            prev_idx = valid_data.index[i-1]
            curr_idx = valid_data.index[i]
            
            prev_irr = valid_data.iloc[i-1]['Irr']
            curr_irr = valid_data.iloc[i]['Irr']
            prev_p = valid_data.iloc[i-1]['P']
            curr_p = valid_data.iloc[i]['P']
            
            # Additional safety check: skip only if BOTH timestamps have very low Irr values
            # Allow cases where one value is high (like your anomaly: 15→288)
            if prev_irr < self.min_irr_threshold and curr_irr < self.min_irr_threshold:
                continue
            
            # Calculate percentage changes
            irr_change = (curr_irr - prev_irr) / prev_irr
            p_change = (curr_p - prev_p) / prev_p
            
            # Skip if Irr change is very small (< 1%) to avoid division by near-zero
            if abs(irr_change) < 0.01:
                continue
            
            # Calculate proportion: P_change / Irr_change
            if irr_change != 0:
                proportion = abs(p_change) / abs(irr_change)
            else:
                proportion = 1.0  # If no Irr change, assume perfect correlation
            
            # Check if proportion is below minimum threshold
            if proportion < self.min_proportion:
                anomaly = {
                    "timestamp": valid_data.iloc[i]['_time'],
                    "type": "irr_p_correlation_mismatch",
                    "irr_change_pct": irr_change * 100,
                    "p_change_pct": p_change * 100,
                    "proportion": proportion * 100,  # Als Prozent
                    "min_proportion_pct": self.min_proportion * 100,
                    "irr_value": curr_irr,
                    "p_value": curr_p,
                    "prev_irr": prev_irr,
                    "prev_p": prev_p,
                    "row_index": curr_idx,
                    "description": f"P changed only {proportion*100:.1f}% relative to Irr change (min required: {self.min_proportion*100:.1f}%)"
                }
                anomalies.append(anomaly)
        
        # DEBUG: Show detected anomalies with details
        if anomalies:
            logger.info(f"DEBUG: First 10 detected correlation anomalies:")
            for i, anomaly in enumerate(anomalies[:10]):
                timestamp = anomaly["timestamp"]
                irr_change = anomaly["irr_change_pct"]
                p_change = anomaly["p_change_pct"]
                proportion = anomaly["proportion"]
                min_proportion = anomaly["min_proportion_pct"]
                irr_value = anomaly["irr_value"]
                p_value = anomaly["p_value"]
                prev_irr = anomaly["prev_irr"]
                prev_p = anomaly["prev_p"]
                description = anomaly["description"]
                logger.info(f"  Anomaly {i+1}: {timestamp}")
                logger.info(f"    Previous: Irr={prev_irr:.1f} W/m², P={prev_p:.1f} W")
                logger.info(f"    Current:  Irr={irr_value:.1f} W/m², P={p_value:.1f} W")
                logger.info(f"    Changes:  Irr={irr_change:.1f}%, P={p_change:.1f}%")
                logger.info(f"    Proportion: P/Irr = {proportion:.1f}% (min required: {min_proportion:.1f}%)")
                logger.info(f"    Reason: {description}")
        
        logger.info(f"Detected {len(anomalies)} correlation anomalies")
        return anomalies
    
    
    def remove_correlation_anomalies(self, data: pd.DataFrame, anomalies: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Remove correlation anomaly rows from the data.
        
        Args:
            data: Input DataFrame
            anomalies: List of detected anomalies
            
        Returns:
            DataFrame with anomalies removed
        """
        if not anomalies:
            return data
        
        # Get row indices to remove - only keep indices that exist in the current DataFrame
        anomaly_indices = [anomaly["row_index"] for anomaly in anomalies]
        valid_indices = [idx for idx in anomaly_indices if idx in data.index]
        
        if not valid_indices:
            logger.info("No valid correlation anomaly indices found in current data")
            return data
        
        # Remove anomaly rows
        original_count = len(data)
        cleaned_data = data.drop(index=valid_indices).copy()
        removed_count = original_count - len(cleaned_data)
        
        # DEBUG: Show first 20 removed correlation anomalies with timestamp and values
        if anomalies:
            logger.info(f"DEBUG: First 20 removed correlation anomalies:")
            for i, anomaly in enumerate(anomalies[:20]):
                row_idx = anomaly["row_index"]
                anomaly_type = anomaly["type"]
                timestamp = anomaly["timestamp"]
                irr_value = anomaly.get("irr_value", "N/A")
                p_value = anomaly.get("p_value", "N/A")
                description = anomaly.get("description", "No description")
                logger.info(f"  Anomaly {i+1}: Row {row_idx}, Time {timestamp}, Type {anomaly_type}")
                logger.info(f"    Irr: {irr_value}, P: {p_value}")
                logger.info(f"    Reason: {description}")
        
        logger.info(f"Removed {removed_count} correlation anomaly records")
        logger.info(f"Data reduced from {original_count} to {len(cleaned_data)} records")
        
        return cleaned_data
    
    def detect_correlation_anomalies_timestamp_based(self, data: pd.DataFrame, dataset_name: str) -> List[Dict[str, Any]]:
        """
        Detect correlation anomalies and return timestamp-based anomaly information.
        This is specifically for averaged/aggregated data where row indices don't match original data.
        
        Args:
            data: DataFrame with averaged/aggregated data
            dataset_name: Name of the dataset
            
        Returns:
            List of anomaly dictionaries with timestamp information
        """
        anomalies = []
        if 'Irr' not in data.columns or 'P' not in data.columns:
            logger.warning("Irr or P columns not found for timestamp-based correlation validation")
            return anomalies
        if '_time' not in data.columns:
            logger.warning("_time column not found for timestamp-based correlation validation")
            return anomalies
        
        logger.info("Starting timestamp-based correlation anomaly detection...")
        data_sorted = data.sort_values('_time').reset_index(drop=True)
        valid_data = data_sorted[
            (data_sorted['Irr'] > 0) &
            (data_sorted['P'] > 0)
        ].copy()
        
        if len(valid_data) < 2:
            logger.warning("Not enough valid data points for timestamp-based correlation validation")
            return anomalies
        
        logger.info(f"Comparing {len(valid_data)} valid data points for timestamp-based correlation anomalies...")

        for i in range(1, len(valid_data)):
            prev_irr = valid_data.iloc[i-1]['Irr']
            curr_irr = valid_data.iloc[i]['Irr']
            prev_p = valid_data.iloc[i-1]['P']
            curr_p = valid_data.iloc[i]['P']
            curr_timestamp = valid_data.iloc[i]['_time']

            # Additional safety check: skip only if BOTH timestamps have very low Irr values
            if prev_irr < self.min_irr_threshold and curr_irr < self.min_irr_threshold:
                continue

            irr_change = (curr_irr - prev_irr) / prev_irr
            p_change = (curr_p - prev_p) / prev_p

            if abs(irr_change) < 0.01: # Skip if Irr change is very small
                continue

            proportion = abs(p_change) / abs(irr_change) if irr_change != 0 else 1.0

            if proportion < self.min_proportion:
                anomaly = {
                    "timestamp": curr_timestamp,
                    "type": "irr_p_correlation_mismatch",
                    "irr_change_pct": irr_change * 100,
                    "p_change_pct": p_change * 100,
                    "proportion": proportion * 100,
                    "min_proportion_pct": self.min_proportion * 100,
                    "irr_value": curr_irr,
                    "p_value": curr_p,
                    "prev_irr": prev_irr,
                    "prev_p": prev_p,
                    "description": f"P changed only {proportion*100:.1f}% relative to Irr change (min required: {self.min_proportion*100:.1f}%)"
                }
                anomalies.append(anomaly)

        return anomalies
    
    def remove_correlation_anomalies_timestamp_based(self, data: pd.DataFrame, anomalies: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Remove correlation anomalies from data based on timestamps.
        This is specifically for averaged/aggregated data.
        
        Args:
            data: DataFrame with averaged/aggregated data
            anomalies: List of anomaly dictionaries with timestamp information
            
        Returns:
            DataFrame with anomalies removed
        """
        if not anomalies:
            logger.info("No timestamp-based correlation anomalies to remove")
            return data
        
        if '_time' not in data.columns:
            logger.warning("No '_time' column found for timestamp-based anomaly removal")
            return data
        
        # Extract timestamps to remove
        anomaly_timestamps = [anomaly['timestamp'] for anomaly in anomalies]
        
        # Remove rows with matching timestamps
        data_cleaned = data[~data['_time'].isin(anomaly_timestamps)].copy()
        
        removed_count = len(data) - len(data_cleaned)
        logger.info(f"Removed {removed_count} timestamp-based correlation anomaly records")
        logger.info(f"Data reduced from {len(data)} to {len(data_cleaned)} records")
        
        return data_cleaned
