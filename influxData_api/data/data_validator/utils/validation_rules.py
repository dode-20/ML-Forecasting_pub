"""
Validation rules and configuration for DataValidator.
"""

from typing import Dict, Any


class ValidationRules:
    """
    Contains validation rules and configuration parameters.
    """
    
    # Define realistic value ranges for PV data (based on typical PV systems)
    VALUE_RANGES = {
        # Temperature ranges (°C)
        "Temp": {"min": -20, "max": 105},  # PV module operating temperature
        "AmbTemp": {"min": -20, "max": 60},  # Ambient temperature
        
        # Humidity ranges (%)
        "AmbHmd": {"min": 0, "max": 100},
        
        # Irradiance ranges (W/m²)
        "Irr": {"min": 0, "max": 1200},  # Maximum solar irradiance
        
        # Electrical parameters
        "Voltage": {"min": 0, "max": 1000},  # Module voltage (V)
        "Current": {"min": -0.1, "max": 20},    # Module current (A)
        "Power": {"min": 0, "max": 500},     # Module power (W)
        
        # Mapped column names
        "U": {"min": 0, "max": 1000},  # Voltage (InfluxDB mapping)
        "I": {"min": -0.1, "max": 20},    # Current (InfluxDB mapping)
        "P": {"min": 0, "max": 300},   # Power (InfluxDB mapping) - Silicon modules
        "T": {"min": -20, "max": 105},  # Temperature (InfluxDB mapping)
        "G": {"min": 0, "max": 1200},  # Irradiance (InfluxDB mapping)
    }
    
    # Perovskite-specific power limits (lower than silicon)
    PEROVSKITE_VALUE_RANGES = {
        "P": {"min": 0, "max": 130},   # Power limit for Perovskite modules (W)
    }
    
    # Perovskite-specific exclusion periods (hardcoded)
    PEROVSKITE_EXCLUSION_PERIODS = [
        {"start": "2024-12-06", "end": "2024-12-13", "reason": "Maintenance period 1"},
        {"start": "2025-05-03", "end": "2025-05-13", "reason": "Maintenance period 2"},
        {"start": "2025-06-26", "end": "2025-06-29", "reason": "Maintenance period 3"},
    ]
    
    # Outlier detection configuration for extended avg mode
    EXTENDED_OUTLIER_CONFIG = {
        "z_score_threshold": 3.0,      # Standard deviation threshold
        "iqr_multiplier": 1.5,         # IQR multiplier for box plot method
        "isolation_forest_contamination": 0.1,  # Expected fraction of outliers
        "lof_contamination": 0.1,      # Local Outlier Factor contamination
        "rolling_window": 24,          # Hours for rolling statistics
        "use_global_stats": False,     # Use module-specific statistics
    }
    
    # Outlier detection configuration for standard mode
    STANDARD_OUTLIER_CONFIG = {
        "z_score_threshold": 4.0,      # Higher threshold to reduce false positives
        "use_global_stats": True,      # Use global statistics instead of module-specific
    }
    
    # Module failure detection parameters (relative to other modules)
    FAILURE_CONFIG = {
        "deviation_threshold": 3.0,    # Standard deviations from module mean
        "min_modules_for_comparison": 3,  # Minimum modules needed for comparison
        "consecutive_hours_threshold": 6,  # Hours of deviation to flag as failure
        "correlation_threshold": 0.3,   # Minimum correlation with other modules
        "zero_tolerance_hours": 2,     # Hours of zero values before flagging (allows for night)
    }
    
    # Correlation validation parameters (Irr-P relationship) - Simplified
    CORRELATION_CONFIG = {
        "min_proportion": 0.05,    # 10% - P change must be at least 10% of Irr change
        "min_irr_threshold": 100, 
    }
    
    @classmethod
    def get_value_ranges(cls, module_type: str = "silicon") -> Dict[str, Dict[str, float]]:
        """
        Get value ranges for the specified module type.
        
        Args:
            module_type: Module type ('silicon' or 'perovskite')
            
        Returns:
            Dictionary of value ranges
        """
        ranges = cls.VALUE_RANGES.copy()
        
        if module_type == "perovskite":
            # Override with Perovskite-specific ranges
            ranges.update(cls.PEROVSKITE_VALUE_RANGES)
        
        return ranges
    
    @classmethod
    def get_outlier_config(cls, extended_mode: bool = False) -> Dict[str, Any]:
        """
        Get outlier detection configuration.
        
        Args:
            extended_mode: Whether to use extended mode configuration
            
        Returns:
            Dictionary of outlier detection parameters
        """
        if extended_mode:
            return cls.EXTENDED_OUTLIER_CONFIG.copy()
        else:
            return cls.STANDARD_OUTLIER_CONFIG.copy()
    
    @classmethod
    def get_failure_config(cls) -> Dict[str, Any]:
        """
        Get module failure detection configuration.
        
        Returns:
            Dictionary of failure detection parameters
        """
        return cls.FAILURE_CONFIG.copy()
