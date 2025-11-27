"""
Quantile scaling module for DataValidator.

This module provides functions for calculating and applying quantile scaling
to power values, creating normalized power columns.
"""

from .scaling_calculator import QuantileScalingCalculator

__all__ = ['QuantileScalingCalculator']
