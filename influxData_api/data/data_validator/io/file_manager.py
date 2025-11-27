"""
File management for DataValidator.
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class FileManager:
    """
    Manages file operations and directory structure for DataValidator.
    """
    
    def __init__(self, output_dir: str, validation_mode: str = "training"):
        """
        Initialize FileManager.
        
        Args:
            output_dir: Base output directory
            validation_mode: 'training' or 'forecast'
        """
        self.validation_mode = validation_mode
        
        # Create timestamped subdirectory for this validation run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Adjust output directory based on validation mode
        if validation_mode == "forecast":
            # For forecast: results/forecast_data/cleanData/validation_run_YYYYMMDD_HHMMSS
            self.output_dir = Path("results/forecast_data/cleanData") / f"validation_run_{timestamp}"
        else:
            # For training: use provided output_dir or default
            self.output_dir = Path(output_dir) / f"validation_run_{timestamp}"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Validation results will be saved to: {self.output_dir}")
        logger.info(f"Validation mode: {validation_mode}")
    
    def save_validation_results(self, dataset_name: str, results: Dict[str, Any]):
        """
        Save validation results to files.
        
        Args:
            dataset_name: Name of the dataset
            results: Validation results dictionary
        """
        # Save summary report
        summary_path = self.output_dir / f"{dataset_name}_validation_summary.json"
        import json
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Validation summary saved: {summary_path}")
    
    def get_output_path(self, dataset_name: str, module_type: str, mode: str = "standard") -> Path:
        """
        Get the output path for cleaned data.
        
        Args:
            dataset_name: Name of the dataset
            module_type: Module type ('silicon' or 'perovskite')
            mode: Mode ('standard' or 'extended_avg')
            
        Returns:
            Path: Output path for the cleaned data
        """
        if self.validation_mode == "forecast":
            output_dir = Path("results/forecast_data/cleanData")
            filename = "latest_forecast_input_clean.csv"
        else:
            # For training: save to module-type specific directory
            if module_type == "silicon":
                output_dir = Path("results/training_data/Silicon/cleanData")
            elif module_type == "perovskite":
                output_dir = Path("results/training_data/Perovskite/cleanData")
            else:
                raise ValueError(f"Unknown module type: {module_type}")
            
            if mode == "extended_avg":
                filename = f"{dataset_name}_03_extended_avg_{module_type}.csv"
            else:
                filename = f"{dataset_name}_03_avg_{module_type}.csv"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / filename
    
    def get_scaling_factors_path(self, dataset_name: str, module_type: str) -> Path:
        """
        Get the path for scaling factors file.
        
        Args:
            dataset_name: Name of the dataset
            module_type: Module type
            
        Returns:
            Path: Path for scaling factors file
        """
        if self.validation_mode == "forecast":
            output_dir = Path("results/forecast_data/cleanData")
        else:
            if module_type == "silicon":
                output_dir = Path("results/training_data/Silicon/cleanData")
            elif module_type == "perovskite":
                output_dir = Path("results/training_data/Perovskite/cleanData")
            else:
                raise ValueError(f"Unknown module type: {module_type}")
            
            # Extract date range from dataset name
            date_range = dataset_name.split("_test_")[0] if "_test_" in dataset_name else "unknown"
            output_dir = output_dir / date_range
        
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / f"{dataset_name}_{module_type}_scaling_factors.csv"
