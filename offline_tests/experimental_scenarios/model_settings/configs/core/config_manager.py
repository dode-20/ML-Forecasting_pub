#!/usr/bin/env python3
"""
Config Manager for Hyperparameter Analysis

This module handles loading and managing configuration files for hyperparameter experiments.

"""

import json
from pathlib import Path
from typing import Dict, Any, List


class ConfigManager:
    """Manages configuration files for hyperparameter analysis"""
    
    def __init__(self, base_config_file: str):
        """
        Initialize ConfigManager
        
        Args:
            base_config_file: Name of the base configuration file (in configs/ directory)
        """
        self.configs_dir = Path(__file__).parent.parent / "configs"
        self.base_config_file = base_config_file
        self.base_config = self._load_base_config()
    
    def _load_base_config(self) -> Dict[str, Any]:
        """Load base configuration file"""
        config_path = self.configs_dir / self.base_config_file
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def generate_parameter_configs(self, parameter_name: str, parameter_values: List[Any], experiment_settings: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate configurations for different parameter values
        
        Args:
            parameter_name: Name of the parameter to vary
            parameter_values: List of values to test for this parameter
            experiment_settings: Additional experiment settings
            
        Returns:
            List of configuration dictionaries
        """
        configs = []
        
        for value in parameter_values:
            # Create a copy of base config
            config = self.base_config.copy()
            
            # Update the specific parameter
            config[parameter_name] = value
            
            # Update model name to include parameter value
            config['model_name'] = f"{self.base_config['model_name']}_{parameter_name}_{value}"
            
            # Add experiment settings
            config.update(experiment_settings)
            
            configs.append(config)
        
        return configs
