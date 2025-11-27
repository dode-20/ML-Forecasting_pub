"""
Data processing utilities for DataValidator.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Handles data processing operations like module name unification, 
    time feature addition, and data transformations.
    """
    
    def __init__(self):
        """Initialize DataProcessor."""
        pass
    
    def unify_module_names(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Unify module names: Convert endings like '-1', '-2', etc. to '_1', '_2', etc.
        This ensures that modules renamed in the database (e.g., 'Atersa-1' to 'Atersa_1')
        are handled consistently.
        
        Args:
            data: Input DataFrame with 'Name' column
            
        Returns:
            DataFrame with unified module names
        """
        if 'Name' not in data.columns:
            return data
        
        data = data.copy()
        # Replace '-' with '_' in module names for consistency
        data['Name'] = data['Name'].str.replace('-', '_')
        
        return data
    
    def add_time_features(self, data: pd.DataFrame, time_features: List[str]) -> pd.DataFrame:
        """
        Add time-based features to the dataset.
        
        Args:
            data: Input DataFrame with '_time' column
            time_features: List of time features to add
            
        Returns:
            DataFrame with added time features
        """
        if '_time' not in data.columns:
            logger.warning("No '_time' column found, skipping time features")
            return data
        
        data = data.copy()
        data['_time'] = pd.to_datetime(data['_time'], errors='coerce')
        
        if 'day_of_year' in time_features:
            data['day_of_year'] = data['_time'].dt.dayofyear
        if 'month' in time_features:
            data['month'] = data['_time'].dt.month
        if 'weekday' in time_features:
            data['weekday'] = data['_time'].dt.weekday
        if 'hour' in time_features:
            data['hour'] = data['_time'].dt.hour
        if 'minute' in time_features:
            data['minute'] = data['_time'].dt.minute
        
        return data
    
    def get_module_type_from_name(self, name: str) -> str:
        """
        Determine module type from module name.
        
        Args:
            name: Module name
            
        Returns:
            Module type ('silicon' or 'perovskite')
        """
        if pd.isna(name):
            return 'silicon'  # Default fallback
        
        name_lower = str(name).lower()
        if 'perovskite' in name_lower:
            return 'perovskite'
        else:
            return 'silicon'
    
    def remove_perovskite_exclusion_periods(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove data from Perovskite exclusion periods.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with exclusion periods removed
        """
        # Perovskite-specific exclusion periods (hardcoded)
        exclusion_periods = [
            {"start": "2024-12-06", "end": "2024-12-13", "reason": "Maintenance period 1"},
            {"start": "2025-05-03", "end": "2025-05-13", "reason": "Maintenance period 2"},
            {"start": "2025-06-26", "end": "2025-06-29", "reason": "Maintenance period 3"},
        ]
        
        if '_time' not in data.columns:
            return data
        
        data = data.copy()
        data['_time'] = pd.to_datetime(data['_time'], errors='coerce')
        
        original_count = len(data)
        
        for period in exclusion_periods:
            start_date = pd.to_datetime(period['start'])
            end_date = pd.to_datetime(period['end'])
            
            # Remove data within exclusion period
            mask = (data['_time'] >= start_date) & (data['_time'] <= end_date)
            data = data[~mask]
        
        removed_count = original_count - len(data)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} records from Perovskite exclusion periods")
        
        return data
    
    def aggregate_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate duplicate timestamps by taking the mean of numeric columns.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with duplicates aggregated
        """
        if data.empty:
            return data
        
        # Define numeric columns for aggregation
        numeric_columns = ['Temp', 'U', 'I', 'P', 'P_normalized', 'AmbTemp', 'AmbHmd', 'Irr']
        available_numeric_columns = [col for col in numeric_columns if col in data.columns]
        
        # Group by timestamp and module, aggregate numeric columns
        agg_dict = {col: 'mean' for col in available_numeric_columns}
        
        # For non-numeric columns, take the first value
        other_columns = [col for col in data.columns if col not in available_numeric_columns]
        for col in other_columns:
            agg_dict[col] = 'first'
        
        # Group by '_time' and 'Name' (module name)
        grouped = data.groupby(['_time', 'Name']).agg(agg_dict).reset_index()
        
        logger.info(f"Aggregated duplicates: {len(data)} -> {len(grouped)} records")
        
        return grouped
