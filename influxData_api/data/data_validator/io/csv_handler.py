"""
CSV handling utilities for DataValidator.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class CSVHandler:
    """
    Handles CSV file operations for DataValidator.
    """
    
    @staticmethod
    def load_csv(file_path: str, **kwargs) -> pd.DataFrame:
        """
        Load CSV file with proper error handling.
        
        Args:
            file_path: Path to CSV file
            **kwargs: Additional arguments for pd.read_csv
            
        Returns:
            Loaded DataFrame
            
        Raises:
            FileNotFoundError: If file doesn't exist
            pd.errors.EmptyDataError: If file is empty
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        
        try:
            df = pd.read_csv(file_path, **kwargs)
            logger.info(f"Successfully loaded CSV: {len(df)} rows, {len(df.columns)} columns")
            return df
        except pd.errors.EmptyDataError:
            logger.error(f"CSV file is empty: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading CSV file {file_path}: {e}")
            raise
    
    @staticmethod
    def save_csv(data: pd.DataFrame, file_path: str, **kwargs) -> None:
        """
        Save DataFrame to CSV file.
        
        Args:
            data: DataFrame to save
            file_path: Path to save CSV file
            **kwargs: Additional arguments for to_csv
        """
        file_path = Path(file_path)
        
        # Create directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            data.to_csv(file_path, index=False, **kwargs)
            logger.info(f"Successfully saved CSV: {file_path}")
        except Exception as e:
            logger.error(f"Error saving CSV file {file_path}: {e}")
            raise
    
    @staticmethod
    def process_timestamps(data: pd.DataFrame) -> pd.DataFrame:
        """
        Process timestamp columns in the DataFrame.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with processed timestamps
        """
        data = data.copy()
        
        if "_time" in data.columns:
            # Remove timezone offsets before conversion
            if data["_time"].dtype == 'object':
                data["_time"] = data["_time"].str.replace(r'[+-]\d{2}:\d{2}$|Z$', '', regex=True)
            
            # Convert to datetime
            data["_time"] = pd.to_datetime(data["_time"], errors='coerce')
        
        return data
    
    @staticmethod
    def validate_csv_structure(data: pd.DataFrame, required_columns: list = None) -> bool:
        """
        Validate CSV structure and required columns.
        
        Args:
            data: DataFrame to validate
            required_columns: List of required column names
            
        Returns:
            True if valid, False otherwise
        """
        if data.empty:
            logger.error("DataFrame is empty")
            return False
        
        if required_columns:
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return False
        
        return True
