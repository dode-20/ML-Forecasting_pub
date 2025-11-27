"""
Range validation methods for DataValidator.
"""

import pandas as pd
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class RangeValidator:
    """
    Handles range validation for PV data.
    """
    
    def __init__(self, value_ranges: Dict[str, Dict[str, float]]):
        """
        Initialize RangeValidator.
        
        Args:
            value_ranges: Dictionary defining min/max values for each column
        """
        self.value_ranges = value_ranges
    
    def validate_value_ranges(self, data: pd.DataFrame, columns: List[str]) -> List[Dict[str, Any]]:
        """
        Validate that values are within realistic ranges.
        
        Args:
            data: Input DataFrame
            columns: List of columns to validate
            
        Returns:
            List of range violations
        """
        violations = []
        
        for column in columns:
            if column not in data.columns:
                continue
                
            if column not in self.value_ranges:
                logger.warning(f"No range definition found for column: {column}")
                continue
            
            min_val = self.value_ranges[column]["min"]
            max_val = self.value_ranges[column]["max"]
            
            # Find violations
            violation_mask = (data[column] < min_val) | (data[column] > max_val)
            violation_indices = data[violation_mask].index.tolist()
            
            if len(violation_indices) > 0:
                for idx in violation_indices:
                    violation = {
                        "timestamp": data.loc[idx, "_time"] if "_time" in data.columns else None,
                        "column": column,
                        "value": data.loc[idx, column],
                        "min_allowed": min_val,
                        "max_allowed": max_val,
                        "row_index": idx
                    }
                    violations.append(violation)
        
        if violations:
            logger.info(f"Found {len(violations)} range violations")
        
        return violations
    
    def remove_violation_timestamps(self, data: pd.DataFrame, violations: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Remove timestamps that have range violations.
        
        Args:
            data: Input DataFrame
            violations: List of range violations
            
        Returns:
            DataFrame with violation timestamps removed
        """
        if not violations:
            return data
        
        # Get unique timestamps with violations
        violation_timestamps = set()
        for violation in violations:
            if violation.get("timestamp"):
                violation_timestamps.add(violation["timestamp"])
        
        if not violation_timestamps:
            return data
        
        # Remove rows with violation timestamps
        original_count = len(data)
        cleaned_data = data[~data["_time"].isin(violation_timestamps)].copy()
        removed_count = original_count - len(cleaned_data)
        
        logger.info(f"Removing {removed_count} records with range violations from {len(violation_timestamps)} timestamps")
        logger.info(f"Data reduced from {original_count} to {len(cleaned_data)} records")
        
        return cleaned_data
