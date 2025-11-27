"""
Module failure detection methods for DataValidator.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class ModuleFailureDetector:
    """
    Handles module failure detection by comparing modules relative to each other.
    """
    
    def __init__(self, failure_config: Dict[str, Any]):
        """
        Initialize ModuleFailureDetector.
        
        Args:
            failure_config: Configuration dictionary for failure detection
        """
        self.config = failure_config
    
    def detect_module_failures_relative(self, data: pd.DataFrame, columns: List[str], dataset_name: str) -> List[Dict[str, Any]]:
        """
        Detect module failures by comparing modules relative to each other.
        
        Args:
            data: Input DataFrame
            columns: List of columns to check for failures
            dataset_name: Name of the dataset
            
        Returns:
            List of detected module failures
        """
        failures = []
        
        # Get unique modules
        modules = data['Name'].unique()
        
        if len(modules) < self.config.get("min_modules_for_comparison", 3):
            logger.warning(f"Not enough modules ({len(modules)}) for failure detection. Minimum required: {self.config.get('min_modules_for_comparison', 3)}")
            return failures
        
        for column in columns:
            if column not in data.columns:
                continue
            
            # Group by timestamp to compare modules at the same time
            for timestamp in data['_time'].unique():
                timestamp_data = data[data['_time'] == timestamp]
                
                if len(timestamp_data) < self.config.get("min_modules_for_comparison", 3):
                    continue
                
                # Calculate statistics for this timestamp
                values = timestamp_data[column].dropna()
                if len(values) < 3:
                    continue
                
                mean_val = values.mean()
                std_val = values.std()
                
                if std_val == 0:  # All values are the same
                    continue
                
                # Check each module for deviation
                for _, row in timestamp_data.iterrows():
                    module_name = row['Name']
                    module_value = row[column]
                    
                    if pd.isna(module_value):
                        continue
                    
                    # Calculate Z-score relative to other modules at this timestamp
                    z_score = abs((module_value - mean_val) / std_val)
                    
                    if z_score > self.config.get("deviation_threshold", 3.0):
                        failure = {
                            "timestamp": timestamp,
                            "column": column,
                            "module": module_name,
                            "value": module_value,
                            "mean_value": mean_val,
                            "std_value": std_val,
                            "z_score": z_score,
                            "threshold": self.config.get("deviation_threshold", 3.0),
                            "row_index": row.name
                        }
                        failures.append(failure)
        
        if failures:
            logger.info(f"Detected {len(failures)} potential module failures")
        
        return failures
    
    def remove_module_failures(self, data: pd.DataFrame, failures: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Remove module failure rows from the data.
        
        Args:
            data: Input DataFrame
            failures: List of failure dictionaries
            
        Returns:
            DataFrame with module failures removed
        """
        if not failures:
            return data
        
        # Get row indices to remove
        failure_indices = [failure["row_index"] for failure in failures]
        
        # Remove failure rows
        original_count = len(data)
        cleaned_data = data.drop(index=failure_indices).copy()
        removed_count = original_count - len(cleaned_data)
        
        # DEBUG: Show first 20 removed module failures with timestamp and value
        if failures:
            logger.info(f"DEBUG: First 20 removed module failures:")
            for i, failure in enumerate(failures[:20]):
                row_idx = failure["row_index"]
                column = failure["column"]
                value = failure["value"]
                module_name = failure.get("module_name", "Unknown")
                timestamp = data.loc[row_idx, "_time"] if "_time" in data.columns else "No timestamp"
                logger.info(f"  Failure {i+1}: Row {row_idx}, Time {timestamp}, Module {module_name}, Column {column}, Value {value}")
        
        logger.info(f"Removed {removed_count} module failure records")
        logger.info(f"Data reduced from {original_count} to {len(cleaned_data)} records")
        
        return cleaned_data
