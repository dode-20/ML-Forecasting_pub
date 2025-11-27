#!/usr/bin/env python3
"""
Evaluator Factory

This module provides a factory class for creating the appropriate evaluator
based on the model configuration. It automatically detects whether a model
is one-step or multi-step and returns the corresponding evaluator instance.

"""

from typing import Dict, Any, Type
from .base_evaluator import BaseEvaluator

# Import evaluators with fallback for direct execution
try:
    from ..forecast_methods.one_step_evaluator import OneStepEvaluator
    from ..forecast_methods.multi_step_evaluator import MultiStepEvaluator
except ImportError:
    from forecast_methods.one_step_evaluator import OneStepEvaluator
    from forecast_methods.multi_step_evaluator import MultiStepEvaluator


class EvaluatorFactory:
    """
    Factory class for creating appropriate evaluators based on model configuration
    """
    
    # Registry of available evaluators
    _evaluators: Dict[str, Type[BaseEvaluator]] = {
        'one-step': OneStepEvaluator,
        'multi-step': MultiStepEvaluator,
    }
    
    @classmethod
    def create_evaluator(cls, model, preprocessor, config: Dict[str, Any], model_name: str) -> BaseEvaluator:
        """
        Create the appropriate evaluator based on model configuration
        
        Args:
            model: Loaded LSTM model
            preprocessor: Loaded data preprocessor
            config: Model configuration dictionary
            model_name: Name of the model being evaluated
            
        Returns:
            Appropriate evaluator instance
            
        Raises:
            ValueError: If the model type is not supported
        """
        # Determine model type from configuration
        model_type = cls._determine_model_type(config)
        
        if model_type not in cls._evaluators:
            available_types = list(cls._evaluators.keys())
            raise ValueError(f"Unsupported model type: {model_type}. Available types: {available_types}")
        
        evaluator_class = cls._evaluators[model_type]
        print(f"[INFO] Creating {evaluator_class.__name__} for {model_type} model")
        
        # Log forecast steps for multi-step models
        if model_type == "multi-step":
            forecast_mode = config.get("forecast_mode", {})
            forecast_steps = forecast_mode.get("forecast_steps", 1)
            print(f"[INFO] Forecast steps: {forecast_steps}")
        
        return evaluator_class(model, preprocessor, config, model_name)
    
    @classmethod
    def _determine_model_type(cls, config: Dict[str, Any]) -> str:
        """
        Determine the model type from configuration
        
        Args:
            config: Model configuration dictionary
            
        Returns:
            String indicating the model type ('one-step' or 'multi-step')
        """
        # Check for forecast_mode configuration (new format)
        forecast_mode = config.get("forecast_mode", {})
        if isinstance(forecast_mode, dict):
            mode = forecast_mode.get("mode", "one-step")
            if mode in ["one-step", "multi-step"]:
                return mode
        
        # Check for legacy forecast_horizon_hours configuration
        forecast_horizon_hours = config.get("forecast_horizon_hours")
        if forecast_horizon_hours is not None:
            if forecast_horizon_hours == 1:
                return "one-step"
            else:
                return "multi-step"
        
        # Check for sequence_length vs output_size relationship
        training_settings = config.get("training_settings", {})
        sequence_length = training_settings.get("sequence_length", 288)
        output_size = config.get("output_size", 1)
        
        # If output_size > 1, it's likely a multi-step model
        if output_size > 1:
            return "multi-step"
        
        # Default to one-step
        return "one-step"
    
    @classmethod
    def register_evaluator(cls, model_type: str, evaluator_class: Type[BaseEvaluator]):
        """
        Register a new evaluator type
        
        Args:
            model_type: String identifier for the model type
            evaluator_class: Class that implements BaseEvaluator
        """
        if not issubclass(evaluator_class, BaseEvaluator):
            raise ValueError(f"Evaluator class must inherit from BaseEvaluator")
        
        cls._evaluators[model_type] = evaluator_class
        print(f"[INFO] Registered evaluator for model type: {model_type}")
    
    @classmethod
    def get_available_evaluators(cls) -> list:
        """
        Get list of available evaluator types
        
        Returns:
            List of available model types
        """
        return list(cls._evaluators.keys())
    
    @classmethod
    def get_evaluator_info(cls) -> Dict[str, str]:
        """
        Get information about available evaluators
        
        Returns:
            Dictionary mapping model types to their descriptions
        """
        info = {}
        for model_type, evaluator_class in cls._evaluators.items():
            info[model_type] = evaluator_class.__doc__.split('\n')[0] if evaluator_class.__doc__ else "No description"
        return info
