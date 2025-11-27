"""
Outlier detection methods for DataValidator.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class OutlierDetector:
    """
    Handles outlier detection for PV data using various statistical methods.
    """
    
    def __init__(self, outlier_config: Dict[str, Any]):
        """
        Initialize OutlierDetector.
        
        Args:
            outlier_config: Configuration dictionary for outlier detection
        """
        self.config = outlier_config
    
    def detect_outliers(self, data: pd.DataFrame, columns: List[str], dataset_name: str) -> List[Dict[str, Any]]:
        """
        Detect outliers using configured methods.
        
        Args:
            data: Input DataFrame
            columns: List of columns to check for outliers
            dataset_name: Name of the dataset
            
        Returns:
            List of detected outliers
        """
        outliers = []
        
        if self.config.get("use_global_stats", False):
            logger.info("Using global outlier detection...")
            return self._detect_outliers_global(data, columns)
        
        # Module-specific detection (more accurate but slower)
        logger.info("Using module-specific outlier detection...")
        z_thresh = self.config.get("z_score_threshold", 3.0)
        
        for column in columns:
            if column not in data.columns:
                continue
            
            # Group by module for module-specific detection
            for module_name in data['Name'].unique():
                module_data = data[data['Name'] == module_name]
                
                if len(module_data) < 10:  # Skip if too few data points
                    continue
                
                # Z-score method
                z_scores = np.abs((module_data[column] - module_data[column].mean()) / module_data[column].std())
                outlier_mask = z_scores > z_thresh
                
                if outlier_mask.any():
                    outlier_indices = module_data[outlier_mask].index.tolist()
                    for idx in outlier_indices:
                        outlier = {
                            "timestamp": module_data.loc[idx, "_time"] if "_time" in module_data.columns else None,
                            "column": column,
                            "value": module_data.loc[idx, column],
                            "module": module_name,
                            "method": "z_score",
                            "z_score": z_scores.loc[idx],
                            "threshold": z_thresh,
                            "row_index": idx
                        }
                        outliers.append(outlier)
        
        if outliers:
            logger.info(f"Detected {len(outliers)} outliers")
        
        return outliers
    
    def _detect_outliers_global(self, data: pd.DataFrame, columns: List[str]) -> List[Dict[str, Any]]:
        """
        Detect outliers using global statistics.
        
        Args:
            data: Input DataFrame
            columns: List of columns to check
            
        Returns:
            List of detected outliers
        """
        outliers = []
        z_thresh = self.config.get("z_score_threshold", 4.0)
        
        for column in columns:
            if column not in data.columns:
                continue
            
            # Global Z-score method
            z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
            outlier_mask = z_scores > z_thresh
            
            if outlier_mask.any():
                outlier_indices = data[outlier_mask].index.tolist()
                for idx in outlier_indices:
                    outlier = {
                        "timestamp": data.loc[idx, "_time"] if "_time" in data.columns else None,
                        "column": column,
                        "value": data.loc[idx, column],
                        "method": "global_z_score",
                        "z_score": z_scores.loc[idx],
                        "threshold": z_thresh,
                        "row_index": idx
                    }
                    outliers.append(outlier)
        
        return outliers
    
    def remove_outliers(self, data: pd.DataFrame, outliers: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Remove outlier rows from the data.
        
        Args:
            data: Input DataFrame
            outliers: List of outlier dictionaries
            
        Returns:
            DataFrame with outliers removed
        """
        if not outliers:
            return data
        
        # Get row indices to remove
        outlier_indices = [outlier["row_index"] for outlier in outliers]
        
        # Remove outlier rows
        original_count = len(data)
        cleaned_data = data.drop(index=outlier_indices).copy()
        removed_count = original_count - len(cleaned_data)
        
        # DEBUG: Show first 20 removed outliers with timestamp and value
        if outliers:
            logger.info(f"DEBUG: First 20 removed outliers:")
            for i, outlier in enumerate(outliers[:20]):
                row_idx = outlier["row_index"]
                column = outlier["column"]
                value = outlier["value"]
                timestamp = data.loc[row_idx, "_time"] if "_time" in data.columns else "No timestamp"
                logger.info(f"  Outlier {i+1}: Row {row_idx}, Time {timestamp}, Column {column}, Value {value}")
        
        logger.info(f"Removed {removed_count} outlier records")
        logger.info(f"Data reduced from {original_count} to {len(cleaned_data)} records")
        
        return cleaned_data
