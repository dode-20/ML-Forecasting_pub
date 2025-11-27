"""
Data Validator Report Generation
===============================

This module provides comprehensive reporting capabilities for the Data Validator.
It generates detailed HTML reports with visualizations and statistics for each validation step.

Classes:
    ValidationReportGenerator: Main class for generating validation reports
    
Functions:
    create_validation_report: Convenience function to create reports
"""

from .validation_report_generator import ValidationReportGenerator, create_validation_report

__all__ = [
    'ValidationReportGenerator',
    'create_validation_report'
]

__version__ = '1.0.0'
