"""
Main DataValidator class - orchestrates the validation process.
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime

from ..modes.standard_mode import StandardMode
from ..modes.extended_avg_mode import ExtendedAvgMode
from ..io.file_manager import FileManager
from ..utils.data_processor import DataProcessor

# Configure logging for the new DataValidator
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger('influxData_api.data_validator_new')


class DataValidator:
    """
    Main DataValidator class that orchestrates the validation process.
    
    This class provides a clean interface to different validation modes
    and delegates the actual work to specialized mode classes.
    """
    
    def __init__(self, 
                 output_dir: str = "training_data/validation_results", 
                 extended_avg_mode: bool = False, 
                 module_type_validation: Optional[str] = None, 
                 raw_data_path: Optional[str] = None,
                 validation_mode: str = "training"):
        """
        Initialize the DataValidator.
        
        Args:
            output_dir (str): Base directory to save validation results and filtered data
            extended_avg_mode (bool): If True, perform full validation (outlier detection, module failure detection) 
                                     on individual modules, then create averages from validated results.
                                     If False (default), use standard mode (simple averaging with basic validation).
            module_type_validation (Optional[str]): 'silicon' or 'perovskite' (required for avg operations)
            raw_data_path (Optional[str]): Path to raw data CSV (required for avg operations)
            validation_mode (str): 'training' or 'forecast' - determines path structure and processing
        """
        self.validation_mode = validation_mode
        self.extended_avg_mode = extended_avg_mode
        self.module_type_validation = module_type_validation
        self.raw_data_path = raw_data_path
        
        # Initialize file manager
        self.file_manager = FileManager(output_dir, validation_mode)
        
        # Initialize data processor
        self.data_processor = DataProcessor()
        
        # Initialize validation results storage
        self.validation_results = {
            "range_violations": [],
            "outliers": [],
            "module_failures": [],
            "summary": {}
        }
        
        # Initialize the appropriate mode
        if extended_avg_mode:
            self.mode = ExtendedAvgMode(
                file_manager=self.file_manager,
                data_processor=self.data_processor,
                module_type_validation=module_type_validation,
                raw_data_path=raw_data_path,
                validation_mode=validation_mode
            )
            logger.info("Extended avg mode enabled - full validation will be performed on individual modules before averaging")
        else:
            self.mode = StandardMode(
                file_manager=self.file_manager,
                data_processor=self.data_processor,
                module_type_validation=module_type_validation,
                raw_data_path=raw_data_path,
                validation_mode=validation_mode
            )
            logger.info("Standard mode enabled - simple averaging with basic validation")
    
    def validate_training_data(self, 
                             data: Optional[Any], 
                             dataset_name: str,
                             features: List[str],
                             outputs: Optional[List[str]] = None,
                             time_features: Optional[List[str]] = None) -> Tuple[Any, Dict[str, Any]]:
        """
        Comprehensive validation of training data.
        
        Args:
            data: Input training data (ignored for avg operations)
            dataset_name: Name of the dataset for output files
            features: List of feature columns to validate
            outputs: List of output columns to validate
            time_features: List of desired time features
            
        Returns:
            Tuple[Any, Dict]: Cleaned data and validation summary
        """
        logger.info(f"Starting validation for dataset: {dataset_name}")
        logger.info(f"Input data shape: {data.shape if data is not None else 'None (avg operations)'}")
        
        # Validate required parameters
        if self.raw_data_path is None:
            raise ValueError("raw_data_path must be provided for avg operations!")
        
        if self.module_type_validation not in ["silicon", "perovskite"]:
            raise ValueError("module_type_validation must be 'silicon' or 'perovskite'!")
        
        # Delegate to the appropriate mode
        cleaned_data, summary = self.mode.validate(
            data=data,
            dataset_name=dataset_name,
            features=features,
            outputs=outputs,
            time_features=time_features
        )
        
        # Store results
        self.validation_results = self.mode.validation_results
        self.validation_results["summary"] = summary
        
        return cleaned_data, summary
    
    def get_validation_report(self) -> str:
        """
        Generate a comprehensive validation report.
        
        Returns:
            str: Validation report
        """
        return self.mode.get_validation_report()
    
    @property
    def output_dir(self) -> Path:
        """Get the output directory."""
        return self.file_manager.output_dir
