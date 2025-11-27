#!/usr/bin/env python3
"""
Configuration management for experimental scenarios.

Handles loading, modification, and generation of model configuration files
for systematic parameter analysis.
"""

import json
import copy
from pathlib import Path
from typing import Dict, Any, List, Iterator
from datetime import datetime

class ConfigManager:
    """Manages model configuration for parameter experiments"""
    
    def __init__(self, base_config_path: str):
        """
        Initialize with base configuration file.
        
        Args:
            base_config_path: Path to base model settings JSON (can be relative to configs/ directory)
        """
        # Handle relative paths within configs directory
        if not Path(base_config_path).is_absolute() and '/' not in base_config_path:
            # Assume it's in the configs directory of the experiment
            configs_dir = Path(__file__).parent.parent / "model_settings" / "configs"
            self.base_config_path = configs_dir / base_config_path
        else:
            self.base_config_path = Path(base_config_path)
            
        self.base_config = self._load_config(self.base_config_path)
        
    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def generate_parameter_configs(self, parameter_name: str, values: List[Any], experiment_settings: Dict[str, Any] = None) -> Iterator[Dict[str, Any]]:
        """
        Generate configurations with parameter variations.
        
        Args:
            parameter_name: Name of parameter to vary
            values: List of values to test
            experiment_settings: Additional experiment settings (e.g., override_module_type)
            
        Yields:
            Modified configuration dictionaries
        """
        for value in values:
            config = copy.deepcopy(self.base_config)
            
            # Apply experiment setting overrides (if any)
            if experiment_settings:
                if "override_module_type" in experiment_settings:
                    config["module_type"] = experiment_settings["override_module_type"]
            
            # Handle nested parameters (e.g., "forecast_mode.forecast_steps")
            if '.' in parameter_name:
                keys = parameter_name.split('.')
                current = config
                for key in keys[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                current[keys[-1]] = value
            else:
                config[parameter_name] = value
            
            # Update model name to include parameter variation
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config["model_name"] = f"{self.base_config['model_name']}_{parameter_name}_{value}_{timestamp}"
            
            yield config
    
    def save_config(self, config: Dict[str, Any], output_path: Path) -> None:
        """Save configuration to file"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=4)
    
    def get_config_summary(self, config: Dict[str, Any]) -> str:
        """Generate human-readable config summary"""
        summary = []
        summary.append(f"Model: {config.get('model_name', 'unknown')}")
        summary.append(f"Module Type: {config.get('module_type', 'unknown')}")
        summary.append(f"Epochs: {config.get('epochs', 'unknown')}")
        summary.append(f"Batch Size: {config.get('batch_size', 'unknown')}")
        summary.append(f"Learning Rate: {config.get('learning_rate', 'unknown')}")
        summary.append(f"Hidden Size: {config.get('hidden_size', 'unknown')}")
        summary.append(f"Sequence Length: {config.get('sequence_length', 'unknown')}")
        
        return "\n".join(summary)
