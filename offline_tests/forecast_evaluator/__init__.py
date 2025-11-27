#!/usr/bin/env python3
"""
Forecast Evaluator Package

This package provides a modular and extensible framework for evaluating
LSTM models with different forecasting approaches.

Components:
- BaseEvaluator: Abstract base class for all evaluators
- OneStepEvaluator: Iterative one-step-ahead evaluation
- MultiStepEvaluator: Direct multi-step-ahead evaluation (to be implemented)
- EvaluatorFactory: Factory for creating appropriate evaluators
- ForecastEvaluator: Main orchestrator class

"""

from .base import BaseEvaluator, EvaluatorFactory
from .forecast_methods import OneStepEvaluator, MultiStepEvaluator
from .forecast_evaluator import ForecastEvaluator

__all__ = [
    'BaseEvaluator',
    'OneStepEvaluator',
    'MultiStepEvaluator',
    'EvaluatorFactory',
    'ForecastEvaluator'
]