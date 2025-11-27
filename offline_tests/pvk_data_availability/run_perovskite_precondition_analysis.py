#!/usr/bin/env python3
"""
Example script to run Perovskite Precondition Analysis

This script demonstrates how to use the perovskite_data_availability_analysis.py
for comparing Silicon and Perovskite module data availability.

Usage:
    python run_perovskite_precondition_analysis.py

Make sure to update the file paths below to point to your actual clean data files.
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Run the precondition analysis with example paths."""
    
    print("="*60)
    print("PEROVSKITE PRECONDITION ANALYSIS - EXAMPLE RUN")
    print("="*60)
    
    # Define paths to clean data files
    # UPDATE THESE PATHS TO YOUR ACTUAL FILES
    base_path = Path(__file__).parent.parent.parent / "results" / "training_data"
    date_range = "20240901_20250725"  # Update this to your actual date range
    
    silicon_path = base_path / "Silicon" / "cleanData" / date_range / f"{date_range}_test_lstm_model_clean-1h.csv"
    perovskite_path = base_path / "Perovskite" / "cleanData" / date_range / f"{date_range}_test_lstm_model_clean-1h.csv"
    
    print(f"Silicon data path: {silicon_path}")
    print(f"Perovskite data path: {perovskite_path}")
    
    # Check if files exist
    if not silicon_path.exists():
        print(f"ERROR: Silicon data file not found: {silicon_path}")
        print("Please update the silicon_path variable in this script.")
        sys.exit(1)
    
    if not perovskite_path.exists():
        print(f"ERROR: Perovskite data file not found: {perovskite_path}")
        print("Please update the perovskite_path variable in this script.")
        sys.exit(1)
    
    # Run the analysis
    analysis_script = Path(__file__).parent / "perovskite_data_availability_analysis.py"
    
    cmd = [
        sys.executable,
        str(analysis_script),
        "--silicon-path", str(silicon_path),
        "--perovskite-path", str(perovskite_path)
    ]
    
    print(f"\nRunning analysis...")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n" + "="*60)
        print("[OK] ANALYSIS COMPLETED SUCCESSFULLY")
        print("="*60)
        return 0
    except subprocess.CalledProcessError as e:
        print("\n" + "="*60)
        print(f"[FAIL] ANALYSIS FAILED with exit code {e.returncode}")
        print("="*60)
        return 1
    except Exception as e:
        print("\n" + "="*60)
        print(f"[FAIL] UNEXPECTED ERROR: {e}")
        print("="*60)
        return 1

if __name__ == "__main__":
    sys.exit(main())
