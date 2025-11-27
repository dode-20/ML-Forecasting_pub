#!/usr/bin/env python3
"""
Evaluation Modes for Hyperparameter Analysis

This module defines different evaluation modes for hyperparameter experiments.

"""

from typing import Dict, Any


class EvaluationModes:
    """Defines different evaluation modes for hyperparameter analysis"""
    
    MODES = {
        "silicon_only": {
            "module_types": ["silicon"],
            "description": "Evaluate only Silicon modules"
        },
        "cross_technology": {
            "module_types": ["silicon", "perovskite"],
            "description": "Evaluate both Silicon and Perovskite modules"
        }
    }
    
    @classmethod
    def get_mode(cls, mode_name: str) -> Dict[str, Any]:
        """
        Get evaluation mode configuration
        
        Args:
            mode_name: Name of the evaluation mode
            
        Returns:
            Mode configuration dictionary
            
        Raises:
            ValueError: If mode_name is not supported
        """
        if mode_name not in cls.MODES:
            available_modes = list(cls.MODES.keys())
            raise ValueError(f"Unsupported evaluation mode: {mode_name}. Available modes: {available_modes}")
        
        return cls.MODES[mode_name]
