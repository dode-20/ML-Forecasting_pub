#!/usr/bin/env python3
"""
Evaluation modes for experimental scenarios.

Defines different evaluation strategies for Silicon-only optimization 
and cross-technology transferability analysis.
"""

from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class EvaluationMode:
    """Configuration for different evaluation modes"""
    
    name: str
    training_data: str  # "silicon" or "perovskite"
    evaluation_data: List[str]  # ["silicon"] or ["silicon", "perovskite"]
    metrics: List[str]
    focus: str
    report_type: str

class EvaluationModes:
    """Predefined evaluation modes for different analysis types"""
    
    SILICON_ONLY = EvaluationMode(
        name="silicon_only",
        training_data="silicon",
        evaluation_data=["silicon"],
        metrics=["RMSE", "MAE", "R²", "MAPE"],
        focus="model_optimization",
        report_type="silicon_optimization"
    )
    
    CROSS_TECHNOLOGY = EvaluationMode(
        name="cross_technology", 
        training_data="silicon",
        evaluation_data=["silicon", "perovskite"],
        metrics=["RMSE", "MAE", "R²", "MAPE", "transfer_degradation", "relative_error"],
        focus="transferability_assessment",
        report_type="transferability_analysis"
    )
    
    @classmethod
    def get_mode(cls, mode_name: str) -> EvaluationMode:
        """Get evaluation mode by name"""
        if mode_name == "silicon_only":
            return cls.SILICON_ONLY
        elif mode_name == "cross_technology":
            return cls.CROSS_TECHNOLOGY
        else:
            raise ValueError(f"Unknown evaluation mode: {mode_name}")
    
    @classmethod
    def list_modes(cls) -> List[str]:
        """List available evaluation modes"""
        return ["silicon_only", "cross_technology"]
