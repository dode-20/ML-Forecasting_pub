"""
DataValidator - Modular and extensible data validation for PV training datasets.

This package provides comprehensive data validation with a clean, modular structure:
- Core validation logic
- Different validation modes (standard, extended avg)
- Extensible validation methods
- Clean IO handling
- Utility functions

Main entry point: DataValidator class from core.validator
"""

from .core.validator import DataValidator
from .modes.standard_mode import StandardMode
from .modes.extended_avg_mode import ExtendedAvgMode

__version__ = "2.0.0"
__all__ = ["DataValidator", "StandardMode", "ExtendedAvgMode"]
