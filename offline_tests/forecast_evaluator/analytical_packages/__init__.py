"""
Analytical Packages for PV Forecasting Evaluation
================================================

This module contains analytical packages for in-depth analysis of
LSTM-based PV forecasting models, specifically focusing on differences
between Silicon and Perovskite module predictions.

Packages:
- package1_variability_analysis: CV prediction quality analysis
- package2_regime_identification: Operational regime analysis (planned)
- package3_heteroskedasticity: Error variance analysis (planned)
"""

from .package1_variability_analysis import (
    VariabilityAnalysis,
    analyze_cv_prediction_quality
)

from .package2_ramp_rate_analysis import (
    RampRateAnalysis,
    analyze_ramp_rate_prediction_quality
)

from .package3_intermittency_analysis import (
    IntermittencyAnalysis,
    analyze_intermittency_prediction_quality
)

from .package4_power_level_analysis import (
    PowerLevelAnalysis,
    analyze_power_level_prediction_quality
)

__all__ = [
    'VariabilityAnalysis',
    'analyze_cv_prediction_quality',
    'RampRateAnalysis',
    'analyze_ramp_rate_prediction_quality',
    'IntermittencyAnalysis',
    'analyze_intermittency_prediction_quality',
    'PowerLevelAnalysis',
    'analyze_power_level_prediction_quality'
]
