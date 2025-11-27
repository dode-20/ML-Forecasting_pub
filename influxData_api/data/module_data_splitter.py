"""
Module Data Splitter

This script processes a CSV file containing time-series data for multiple solar modules
and separates it into individual CSV files per module. The output files are organized
into folders by module type (silicon or perovskite), and the filenames include the
time range and module name for traceability.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Literal
import shutil


class ModuleDataSplitter:
    """
    A class to split a CSV file with data from multiple solar modules into separate files
    per module, categorized by module type.
    """

    def __init__(self, input_csv: str, output_dir: str) -> None:
        """
        Initialize the splitter with CSV file path and output directory.

        Args:
            input_csv (str): Path to the input CSV file containing module data.
            output_dir (str): Directory where output files will be stored.
        """
        self.input_csv = input_csv
        self.output_dir = output_dir

        # Predefined module classifications by MAC address (expanded)
        self.silicon_modules = {
            "083AF2ACFAC5": "Atersa_1_1",
            "083AF2B8FC71": "Atersa_2_1",
            "083AF2BA8631": "Atersa_3_1",
            "083AF2BA89F9": "Atersa_4_1",
            "240AC4AEAF21": "Atersa_5_1",
            "240AC4AF600D": "Atersa_6_1",
            "8813BF0BE76D": "Sanyo_2_1",
            "94B97EE92EB1": "Sanyo_3_1",
            "94B97EEA8635": "Sanyo_4_1",
            "B4E62D97412A": "Sanyo_5_1",
            "C8C9A3F9F601": "Solon_1_1",
            "C8C9A3F9FAD5": "Solon_1_2",
            "C8C9A3FA3CCD": "Solon_2_1",
            "C8C9A3FA632D": "Solon_2_2",
            "C8C9A3FAABA5": "Solon_3_2",
            "C8C9A3FAAD85": "Sun_Power_1_1",
            "C8C9A3FCC4B1": "Sun_Power_2_1",
            "C8C9A3FCD0A9": "Sun_Power_3_1",
            "C8C9A3FCD2F9": "Sun_Power_4_1",
            "C8C9A3FD06AD": "Sun_Power_5_1"
        }
        self.perovskite_modules = {
            "240AC4AF6795": "Perovskite_1_1",
            "30AEA47390D9": "Perovskite_1_2",
            "30AEA48AE2B5": "Perovskite_1_3",
            "3C8A1FA8F185": "Perovskite_2_1",
            "3C8A1FA8FA85": "Perovskite_2_2",
            "3C8A1FA90BE9": "Perovskite_2_3"
        }

        # Set the filtering category: 'silicon', 'perovskite', or 'both'
        self.allowed_module_type: Literal["silicon", "perovskite", "both"] = "both"

    def load_data(self) -> None:
        """
        Load and parse the CSV file. Converts time column to datetime.
        """
        self.df = pd.read_csv(self.input_csv)
        if "_time" in self.df.columns:
            self.df["_time"] = pd.to_datetime(self.df["_time"])
        if "_start" in self.df.columns:
            self.df["_start"] = pd.to_datetime(self.df["_start"])
        if "_stop" in self.df.columns:
            self.df["_stop"] = pd.to_datetime(self.df["_stop"])

    def split_and_save(self) -> None:
        """
        Split the input data by module, filter by allowed module type,
        and save each module's data to a separate CSV file.
        """

        # Clear target subdirectories
        for subfolder in ["silicon_modules", "perovskite_modules"]:
            folder_path = self.output_dir / subfolder
            if folder_path.exists():
                shutil.rmtree(folder_path)
            folder_path.mkdir(parents=True, exist_ok=True)

        self.load_data()

        all_modules = {
            **self.silicon_modules,
            **self.perovskite_modules
        }

        # Determine which module types to process
        if self.allowed_module_type == "silicon":
            modules_to_process = self.silicon_modules
        elif self.allowed_module_type == "perovskite":
            modules_to_process = self.perovskite_modules
        elif self.allowed_module_type == "both":
            modules_to_process = all_modules
        else:
            modules_to_process = {}

        # Process and export data per module
        for mac, name in modules_to_process.items():
            module_df = self.df[
                # (self.df["MAC"] == mac) &
                (self.df["Name"] == name)
            ]
            if module_df.empty:
                continue

            start = module_df["_start"].min().strftime("%d-%m-%y")
            end = module_df["_stop"].max().strftime("%d-%m-%y")
            filename = f"{start}_{end}_{name}.csv"

            folder = "silicon_modules" if mac in self.silicon_modules else "perovskite_modules"
            output_path = self.output_dir / folder
            output_path.mkdir(parents=True, exist_ok=True)

            full_path = output_path / filename
            module_df.to_csv(full_path, index=False)


if __name__ == "__main__":
    splitter = ModuleDataSplitter("training_data/lstm/150625-LSTM-Silic_Neu8_data.csv")
    splitter.allowed_module_type = "silicon"  # Options: 'silicon', 'perovskite', 'both'
    splitter.split_and_save()
