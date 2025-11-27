"""
Quantile scaling calculator for power normalization.

This module provides functionality to calculate and apply quantile scaling
factors to power values, creating normalized power columns (P_normalized).
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class QuantileScalingCalculator:
    """
    Handles quantile scaling calculations and applications for power normalization.
    
    This class provides methods to:
    - Calculate scaling factors based on 95th percentile of power values
    - Apply scaling to create P_normalized columns
    - Save scaling factors to CSV files
    """
    
    def __init__(self):
        pass
    
    def calculate_scaling_factors(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate quantile scaling factors for each module's power values.
        
        Args:
            data: Input DataFrame containing power values per module
            
        Returns:
            Dictionary mapping module names to scaling factors
        """
        scaling_factors = {}
        
        if 'P' not in data.columns:
            logger.warning("Power column 'P' not found for scaling factor calculation")
            return scaling_factors
        
        # Check if we have individual modules (Name column) or aggregated data
        if 'Name' in data.columns:
            # Individual modules: calculate scaling factor for each module
            for module_name in data['Name'].unique():
                module_data = data[data['Name'] == module_name]
                power_values = module_data['P'].dropna()
                
                if len(power_values) == 0:
                    logger.warning(f"No power data available for module {module_name}")
                    continue
                
                # Use 95th percentile as scaling factor (robust against outliers)
                scaling_factor = power_values.quantile(0.95)
                
                if scaling_factor > 0:
                    scaling_factors[module_name] = scaling_factor
                    logger.debug(f"Scaling factor for {module_name}: {scaling_factor:.2f} W")
                else:
                    logger.warning(f"Invalid scaling factor for {module_name}: {scaling_factor}")
        else:
            # Aggregated data: use a single scaling factor based on all power values
            power_values = data['P'].dropna()
            
            if len(power_values) == 0:
                logger.warning("No power data available for scaling factor calculation")
                return scaling_factors
            
            # Use 95th percentile as scaling factor (robust against outliers)
            scaling_factor = power_values.quantile(0.95)
            
            if scaling_factor > 0:
                scaling_factors['aggregated'] = scaling_factor
                logger.debug(f"Scaling factor for aggregated data: {scaling_factor:.2f} W")
            else:
                logger.warning(f"Invalid scaling factor for aggregated data: {scaling_factor}")
        
        logger.info(f"Calculated scaling factors for {len(scaling_factors)} modules")
        return scaling_factors
    
    def apply_scaling(self, data: pd.DataFrame, scaling_factors: Dict[str, float]) -> pd.DataFrame:
        """
        Apply quantile scaling to power values and add normalized power column.
        
        Args:
            data: Input DataFrame containing power values
            scaling_factors: Dictionary of scaling factors per module
            
        Returns:
            DataFrame with additional 'P_normalized' column
        """
        data_scaled = data.copy()
        
        if 'P' not in data_scaled.columns:
            logger.warning("Power column 'P' not found for scaling")
            return data_scaled
        
        # Initialize normalized power column
        data_scaled['P_normalized'] = np.nan
        
        # Check if we have individual modules (Name column) or aggregated data
        if 'Name' in data_scaled.columns:
            # Individual modules: apply scaling for each module
            for module_name, scaling_factor in scaling_factors.items():
                module_mask = data_scaled['Name'] == module_name
                module_power = data_scaled.loc[module_mask, 'P']
                
                # Normalize: P_normalized = P / scaling_factor
                normalized_power = module_power / scaling_factor
                
                # Clip values to 1.2 to handle outliers
                normalized_power = np.clip(normalized_power, 0, 1.2)
                
                data_scaled.loc[module_mask, 'P_normalized'] = normalized_power
        else:
            # Aggregated data: apply scaling to all power values
            if 'aggregated' in scaling_factors:
                scaling_factor = scaling_factors['aggregated']
                power_values = data_scaled['P']
                
                # Normalize: P_normalized = P / scaling_factor
                normalized_power = power_values / scaling_factor
                
                # Clip values to 1.2 to handle outliers
                normalized_power = np.clip(normalized_power, 0, 1.2)
                
                data_scaled['P_normalized'] = normalized_power
        
        logger.info(f"Applied quantile scaling to {len(scaling_factors)} modules")
        return data_scaled
    
    def save_scaling_factors(self, scaling_factors: Dict[str, float], dataset_name: str, module_type: str, output_dir: Path):
        """
        Save scaling factors to CSV file.
        
        Args:
            scaling_factors: Dictionary of scaling factors per module
            dataset_name: Dataset name for file naming
            module_type: Module type ('silicon' or 'perovskite')
            output_dir: Output directory for saving files
        """
        if not scaling_factors:
            logger.warning("No scaling factors to save")
            return
        
        # Create scaling factors DataFrame
        scaling_df = pd.DataFrame([
            {'module_name': module_name, 'scaling_factor': factor}
            for module_name, factor in scaling_factors.items()
        ])
        
        # Determine output directory
        date_range = dataset_name.split("_test_")[0]
        final_output_dir = Path("results/training_data") / module_type.capitalize() / "cleanData" / date_range
        final_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save scaling factors
        filename = f"{dataset_name}_{module_type}_scaling_factors.csv"
        filepath = final_output_dir / filename
        scaling_df.to_csv(filepath, index=False)
        
        logger.info(f"Scaling factors saved: {filepath}")
    
    def calculate_and_apply_scaling(self, data: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, float]]:
        """
        Calculate scaling factors and apply them to create P_normalized column.
        
        Args:
            data: Input DataFrame containing power values
            
        Returns:
            Tuple of (scaled_data, scaling_factors)
        """
        logger.info("Calculating scaling factors from validated individual modules...")
        
        # Calculate scaling factors
        scaling_factors = self.calculate_scaling_factors(data)
        
        # Apply scaling to create P_normalized column
        logger.info("Applying quantile scaling to individual modules...")
        scaled_data = self.apply_scaling(data, scaling_factors)
        
        return scaled_data, scaling_factors
