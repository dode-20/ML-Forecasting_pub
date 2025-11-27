from pathlib import Path
from dotenv import load_dotenv
import os
from influxdb_client import InfluxDBClient
import pandas as pd
from typing import Union
import warnings
from influxdb_client.client.warnings import MissingPivotFunction
import numpy as np
from datetime import datetime, timezone
import pytz

# Handle relative import for Docker environment
try:
    from data.module_data_splitter import ModuleDataSplitter
except ImportError:
    # For offline tests, create a simple fallback
    class ModuleDataSplitter:
        def __init__(self):
            pass
        
        def split_data(self, df: pd.DataFrame, validation_split: float = 0.15, shuffle: bool = True) -> tuple:
            """Simple data splitter for offline tests"""
            if shuffle:
                df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            # Interval split
            n = round(1 / validation_split)
            idx = range(len(df))
            val_mask = [(i % n) == 0 for i in idx]
            train_mask = [not mask for mask in val_mask]
            
            train_data = df[train_mask]
            val_data = df[val_mask]
            
            return train_data, val_data

warnings.simplefilter("ignore", MissingPivotFunction)

class influxClient:
    """
    Client for connecting to InfluxDB and querying data.
    Supports both Docker and offline modes.

    Initializes connection parameters from environment variables loaded from a .env file.

    Attributes:
        url (str): InfluxDB URL.
        token (str): Authentication token.
        org (str): Organization name.
        bucket (str): Bucket name.
        client (InfluxDBClient): InfluxDB client instance.
        query_api: Query API interface for running queries.
        mode (str): "docker" or "offline"
    """
    
    # Minimum date for data queries - first data was written on 2024-08-16
    MIN_DATA_DATE = "2024-08-16"
    
    # Modul-Zuordnungstabellen als Klassenattribute
    silicon_modules = {
        "083AF2ACFAC5": "Atersa_1_1",
        "083AF2B8FC71": "Atersa_2_1",
        "083AF2BA8631": "Atersa_3_1",
        "083AF2BA89F9": "Atersa_4_1",
        "240AC4AEAF21": "Atersa_5_1",
        "240AC4AF600D": "Atersa_6_1",
        "8813BF0BE76D": "Sanyo_1_1",
        "8813BF0BE76D": "Sanyo_2_1",
        "94B97EE92EB1": "Sanyo_3_1",
        "94B97EEA8635": "Sanyo_4_1",
        "B4E62D97412A": "Sanyo_5_1",
        "C8C9A3F9F601": "Solon_1_1",
        "C8C9A3F9FAD5": "Solon_1_2",
        "C8C9A3FA3CCD": "Solon_2_1",
        "C8C9A3FA632D": "Solon_2_2",
        "C8C9A3FA632D": "Solon_3_1",
        "C8C9A3FAABA5": "Solon_3_2",
        "C8C9A3FAAD85": "Solon_4_2",
        "C8C9A3FAAD85": "Sun_Power_1_1",
        "C8C9A3FCC4B1": "Sun_Power_2_1",
        "C8C9A3FCD0A9": "Sun_Power_3_1",
        "C8C9A3FCD2F9": "Sun_Power_4_1",
        "C8C9A3FD06AD": "Sun_Power_5_1"
    }
    perovskite_modules = {
        "240AC4AF6795": "Perovskite_1",
        "240AC4AF6795": "Perovskite_1_1",
        "30AEA47390D9": "Perovskite_1_2",
        "30AEA48AE2B5": "Perovskite_1_3",
        "240AC4AF6795": "Perovskite_1_4",
        "240AC4AF6795": "Perovskite_2",
        "3C8A1FA8F185": "Perovskite_2_1",
        "3C8A1FA8FA85": "Perovskite_2_2",
        "3C8A1FA90BE9": "Perovskite_2_3",
        "240AC4AF6795": "Perovskite_3_1",
        "240AC4AF6795": "Perovskite_3_2",
        "240AC4AF6795": "Perovskite_3_3",
        "240AC4AF6795": "Perovskite_4_1",
        "240AC4AF6795": "Perovskite_4_2",
        "240AC4AF6795": "Perovskite_4_3",
    }
    silicon_names = set(silicon_modules.values())
    perovskite_names = set(perovskite_modules.values())

    @staticmethod
    def get_module_type(name):
        if name in influxClient.silicon_names:
            return "silicon"
        elif name in influxClient.perovskite_names:
            return "perovskite"
        else:
            return "unknown"

    def __init__(self, env_path: str = None) -> None:
        """
        Initialize the InfluxDB client by loading environment variables and establishing a connection.

        Args:
            env_path (str): Path to the .env file containing InfluxDB credentials and settings.
        """
        # Detect environment mode
        self.mode = self._detect_environment_mode()
        
        if env_path and os.path.exists(env_path):
            # Load .env file
            load_dotenv(env_path)
        elif self.mode == "docker":
            # In Docker mode, try to load from default location
            default_env_path = Path("/app/.env")
            if default_env_path.exists():
                load_dotenv(default_env_path)
        
        # Access environment variables or use defaults for offline mode
        if self.mode == "offline":
            self.url = os.getenv("INFLUX_URL", "http://localhost:8086")
            self.token = os.getenv("INFLUX_TOKEN")
            self.org = os.getenv("INFLUX_ORG", "admin")
            self.bucket = os.getenv("INFLUX_BUCKET", "pv_data")
        else:
            # Docker mode - use environment variables as before
            self.url = os.getenv("INFLUX_URL")
            self.token = os.getenv("INFLUX_TOKEN")
            self.org = os.getenv("INFLUX_ORG")
            self.bucket = os.getenv("INFLUX_BUCKET")
        
        # Establish connection
        try:
            if self.mode == "offline" and (not self.url or not self.token):
                # Offline mode without valid credentials - use sample data
                self.client = None
                self.query_api = None
                print("Offline mode: Using sample data (no InfluxDB connection)")
            else:
                # Try to connect to InfluxDB (either Docker or offline with credentials)
                if not self.url or not self.token:
                    print("Error: Missing InfluxDB credentials")
                    self.client = None
                    self.query_api = None
                else:
                    # Configure InfluxDB client with extended timeout for large queries
                    self.client = InfluxDBClient(
                        url=self.url, 
                        token=self.token, 
                        org=self.org, 
                        verify_ssl=False,  # Disable SSL verification due to expired certificate
                        timeout=3600  # 60 minutes timeout for very large queries
                    )
                    self.query_api = self.client.query_api()
                    print(f"Connected to InfluxDB at {self.url} with 60-minute timeout")
        except Exception as e:
            if self.mode == "offline":
                print(f"Offline mode: Could not connect to InfluxDB, using sample data: {e}")
                self.client = None
                self.query_api = None
            else:
                print("Error: Could not connect to InfluxDB.")
                print(str(e))
                self.client = None
                self.query_api = None
    
    def _detect_environment_mode(self) -> str:
        """Detect if running in Docker or offline mode"""
        # Check for Docker environment variables
        if os.getenv("DOCKER_ENV") or os.getenv("KUBERNETES_SERVICE_HOST"):
            return "docker"
        
        # Check for offline mode environment variable
        if os.getenv("OFFLINE_MODE") == "true":
            return "offline"
        
        # Check if running in a container (most reliable method)
        if os.path.exists("/.dockerenv"):
            return "docker"
        
        # Check for Docker-specific environment variables that are set in containers
        if os.getenv("HOSTNAME") and os.getenv("HOSTNAME").startswith("ml-forecasting-"):
            return "docker"
        
        # Check for Docker Compose environment variables
        if os.getenv("COMPOSE_PROJECT_NAME") or os.getenv("DOCKER_COMPOSE"):
            return "docker"
        
        # Default to offline mode for safety (only for local development)
        return "offline"
    
    def _validate_date_range(self, start_date: str, end_date: str = None) -> tuple[str, str]:
        """
        Validate and adjust date range to ensure it doesn't go before MIN_DATA_DATE
        
        Args:
            start_date: Requested start date
            end_date: Requested end date (optional)
            
        Returns:
            Tuple of (validated_start_date, validated_end_date)
        """
        # Ensure start_date is not before MIN_DATA_DATE
        if start_date < self.MIN_DATA_DATE:
            print(f"WARNING: Start date {start_date} is before first data date {self.MIN_DATA_DATE}. Adjusting to {self.MIN_DATA_DATE}")
            start_date = self.MIN_DATA_DATE
        
        # If no end_date provided, use current date
        if end_date is None:
            end_date = pd.Timestamp.now().strftime("%Y-%m-%d")
        
        # Ensure end_date is not before start_date
        if end_date < start_date:
            print(f"WARNING: End date {end_date} is before start date {start_date}. Adjusting end date to start date.")
            end_date = start_date
        
        return start_date, end_date
    
    def _generate_sample_data(self, features: list, outputs: list, date_selection: dict = None) -> pd.DataFrame:
        """Generate sample data for testing when InfluxDB is not available"""
        # Determine date range
        if date_selection and isinstance(date_selection, dict):
            mode = date_selection.get("mode", "custom")
            if mode == "custom":
                start_date = date_selection.get("start", "2024-08-16")
                end_date = date_selection.get("end", "2024-08-23")
            else:
                start_date = "2024-08-16"
                end_date = "2024-08-23"
        else:
            start_date = "2024-08-16"
            end_date = "2024-08-23"
        
        # Validate date range
        start_date, end_date = self._validate_date_range(start_date, end_date)
        
        # Create time series
        dates = pd.date_range(start=start_date, end=end_date, freq='H')
        
        # Generate realistic PV data
        np.random.seed(42)  # For reproducible results
        
        data = {}
        
        # Generate features
        for feature in features:
            if feature == 'U':
                data[feature] = 400 + 50 * np.sin(np.pi * dates.hour / 12) + np.random.normal(0, 10, len(dates))
            elif feature == 'I':
                data[feature] = 5 + 3 * np.sin(np.pi * dates.hour / 12) + np.random.normal(0, 0.5, len(dates))
            elif feature == 'P':
                data[feature] = 2000 + 1500 * np.sin(np.pi * dates.hour / 12) + np.random.normal(0, 100, len(dates))
            elif feature == 'T':
                data[feature] = 20 + 15 * np.sin(np.pi * dates.hour / 12) + np.random.normal(0, 2, len(dates))
            elif feature == 'G':
                data[feature] = 800 + 600 * np.sin(np.pi * dates.hour / 12) + np.random.normal(0, 50, len(dates))
            else:
                data[feature] = np.random.normal(0, 1, len(dates))
        
        # Generate outputs
        for output in outputs:
            if output == 'Current':
                data[output] = 5 + 3 * np.sin(np.pi * dates.hour / 12) + np.random.normal(0, 0.3, len(dates))
            elif output == 'Voltage':
                data[output] = 400 + 50 * np.sin(np.pi * dates.hour / 12) + np.random.normal(0, 8, len(dates))
            elif output == 'Power':
                data[output] = 2000 + 1500 * np.sin(np.pi * dates.hour / 12) + np.random.normal(0, 80, len(dates))
            else:
                data[output] = np.random.normal(0, 1, len(dates))
        
        df = pd.DataFrame(data, index=dates)
        
        # Ensure all values are positive
        for col in df.columns:
            df[col] = df[col].abs()
        
        return df

    def query_parkdata_last_xh(self, hours: int) -> pd.DataFrame:
        """
        Query the last X hours of parking data from InfluxDB.

        Runs a Flux query to retrieve the ParkData measurement for the last X hours, pivots the results,
        and combines them into a single DataFrame. The data is also saved to a CSV file.

        Returns:
            pd.DataFrame: DataFrame containing the last X hours of parking data.
        """
        if self.client is None:
            # Generate sample data for offline mode
            return self._generate_sample_data(
                features=['U', 'I', 'P', 'T', 'G'],
                outputs=['Current', 'Voltage', 'Power'],
                date_selection={'mode': 'last', 'value': hours}
            )
        
        # Original Docker mode logic
        timezone_option = self._get_timezone_option()
        query = f'''
        {timezone_option}

        from(bucket:"{self.bucket}")
        |> range(start: -{hours}h)
        |> filter(fn: (r) => r._measurement == "ParkData")
        |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''
        df = self.query_api.query_data_frame(query)
        if isinstance(df, list):
            for i, d in enumerate(df):
                print(f"--- DataFrame {i} ---")
                print(d.head())
        else:
            print(df.head())

        if isinstance(df, list):
            df_combined = pd.concat(df, ignore_index=True)
        else:
            df_combined = df

        # Adjust timezone to German timezone
        df_combined = self._adjust_data_timezone(df_combined)

        # Post-process: Intelligent PV-specific imputation
        #df_combined = self._apply_pv_intelligent_imputation(df_combined)

        # Save to CSV
        df_combined.to_csv("training_data/influx_export.csv", index=False)
        return df_combined

    def query_parkdata_all(self) -> pd.DataFrame:
        """
        Query all available historical ParkData from InfluxDB until now.

        Runs a Flux query to retrieve the complete ParkData measurement, pivots the results,
        and saves them to a CSV file for model training.

        Returns:
            pd.DataFrame: DataFrame containing all historical ParkData.
        """
        if self.client is None:
            # Generate sample data for offline mode
            return self._generate_sample_data(
                features=['U', 'I', 'P', 'T', 'G'],
                outputs=['Current', 'Voltage', 'Power'],
                date_selection={'mode': 'all'}
            )
        
        # Original Docker mode logic - use MIN_DATA_DATE as start
        timezone_option = self._get_timezone_option()
        query = f'''
        {timezone_option}

        from(bucket:"{self.bucket}")
        |> range(start: time(v: "{self.MIN_DATA_DATE}T00:00:00Z"))
        |> filter(fn: (r) => r._measurement == "ParkData")
        |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''
        df = self.query_api.query_data_frame(query)
        if isinstance(df, list):
            df_combined = pd.concat(df, ignore_index=True)
        else:
            df_combined = df

        # Adjust timezone to German timezone
        df_combined = self._adjust_data_timezone(df_combined)

        # Post-process: Intelligent PV-specific imputation
        #df_combined = self._apply_pv_intelligent_imputation(df_combined)

        # Save to CSV
        df_combined.to_csv("lstm_model/data/influx_export_all.csv", index=False)
        return df_combined
    
    def get_module_overview(self) -> pd.DataFrame:
        """
        Retrieve metadata overview of all modules from ParkData.

        Returns:
            pd.DataFrame: DataFrame with module MAC, Name, available fields,
                          status (active in last 24h), first and last data timestamp.
        """
        if self.client is None:
            # Return sample module overview for offline mode
            sample_modules = [
                {
                    "MAC": "00:11:22:33:44:55",
                    "Name": "perovskite_1_1",
                    "Fields": ["U", "I", "P", "T", "G"],
                    "Active (last 24h)": True,
                    "Start Timestamp": pd.Timestamp("2024-08-16"),
                    "Last Timestamp": pd.Timestamp.now()
                },
                {
                    "MAC": "00:11:22:33:44:66",
                    "Name": "sanyo_5_1",
                    "Fields": ["U", "I", "P", "T", "G"],
                    "Active (last 24h)": True,
                    "Start Timestamp": pd.Timestamp("2024-08-16"),
                    "Last Timestamp": pd.Timestamp.now()
                }
            ]
            return pd.DataFrame(sample_modules)
        
        # Original Docker mode logic
        timezone_option = self._get_timezone_option()
        query = f'''
        {timezone_option}

        from(bucket:"{self.bucket}")
        |> range(start: time(v: "{self.MIN_DATA_DATE}T00:00:00Z"))
        |> filter(fn: (r) => r._measurement == "ParkData")
        |> keep(columns: ["_time", "_field", "MAC", "Name"])
        '''
        df = self.query_api.query_data_frame(query)
        if isinstance(df, list):
            df = pd.concat(df, ignore_index=True)

        # Group by MAC + Name
        module_info = []
        now = pd.Timestamp.now(tz="Europe/Berlin")

        for (mac, name), group in df.groupby(["MAC", "Name"]):
            fields = sorted(group["_field"].unique())
            first_time = group["_time"].min()
            last_time = group["_time"].max()
            active = (now - last_time).total_seconds() <= 24 * 3600

            module_info.append({
                "MAC": mac,
                "Name": name,
                "Fields": fields,
                "Active (last 24h)": active,
                "Start Timestamp": first_time,
                "Last Timestamp": last_time if not active else None
            })

        return pd.DataFrame(module_info)
    
    def get_lstm_data(self, dataset_name: str, module_type: str, use_all_modules: str, selected_modules: Union[str, list], features: list, outputs: list = None, date_selection: dict = None) -> pd.DataFrame:
        """
        Constructs and executes a query for training data based on input filters.

        Args:
            dataset_name (str): Name of the dataset for saving CSV.
            model_type (str): Model type filter (not currently used in query).
            module_type (str): Module type filter (e.g., "Silicon", "Perovskite").
            use_all_modules (str): If True, include all modules of this type.
            selected_modules (Union[str, list]): Specific module names to filter if use_all_modules is False.
            features (list): List of input feature names to retrieve.
            outputs (list): List of output feature names to retrieve.
            date_selection (str): Date range selection string, e.g. "last 48h".

        Returns:
            pd.DataFrame: Combined DataFrame of requested fields.
        """

        if outputs is None:
            outputs = []
        
        if self.client is None:
            # Generate sample data for offline mode
            return self._generate_sample_data(features, outputs, date_selection)
        
        # Separate features into module-specific and environmental features
        module_features = [f for f in features if f not in ["AmbTemp", "AmbHmd", "Irr"]]
        env_features = [f for f in features if f in ["AmbTemp", "AmbHmd", "Irr"]]

        # Add outputs to module features (they need to be retrieved from modules)
        all_module_fields = module_features + outputs + env_features

        # Build date range clause with validation
        if isinstance(date_selection, dict):
            mode = date_selection.get("mode", "all").lower()
            if mode == "custom":
                start = date_selection["start"]
                end = date_selection["end"]
                # Validate date range
                start, end = self._validate_date_range(start, end)
                range_clause = f'|> range(start: time(v: "{start}T00:00:00Z"), stop: time(v: "{end}T23:59:59Z"))'
            elif mode == "last":
                try:
                    hours = int(date_selection.get("value", 720))
                    range_clause = f'|> range(start: -{hours}h)'
                except:
                    range_clause = f'|> range(start: time(v: "{self.MIN_DATA_DATE}T00:00:00Z"))'
            else:
                range_clause = f'|> range(start: time(v: "{self.MIN_DATA_DATE}T00:00:00Z"))'
        else:
            range_clause = f'|> range(start: time(v: "{self.MIN_DATA_DATE}T00:00:00Z"))'

        # Query 1: Module-specific data (including outputs)
        if all_module_fields:
            module_fields_filter = " or ".join([f'r._field == "{field}"' for field in all_module_fields])
            
            if not use_all_modules and isinstance(selected_modules, list) and selected_modules:
                module_filter = " or ".join([f'r.Name == "{module}"' for module in selected_modules])
                timezone_option = self._get_timezone_option()
                query1 = f'''
                {timezone_option}

                from(bucket:"{self.bucket}")
                {range_clause}
                |> filter(fn: (r) => r._measurement == "ParkData")
                |> filter(fn: (r) => {module_filter})
                |> filter(fn: (r) => {module_fields_filter})
                |> aggregateWindow(every: 5m, fn: mean, createEmpty: false)
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
                '''
            else:
                timezone_option = self._get_timezone_option()
                query1 = f'''
                {timezone_option}

                from(bucket:"{self.bucket}")
                {range_clause}
                |> filter(fn: (r) => r._measurement == "ParkData")
                |> filter(fn: (r) => {module_fields_filter})
                |> aggregateWindow(every: 5m, fn: mean, createEmpty: false)
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
                '''
            
            df1 = self.query_api.query_data_frame(query1)
            if isinstance(df1, list):
                df1 = pd.concat(df1, ignore_index=True)
        else:
            df1 = pd.DataFrame()

        # Query 2: Environmental data
        if env_features:
            env_fields_filter = " or ".join([f'r._field == "{field}"' for field in env_features])
            
            # Environmental data comes from specific modules
            env_modules = ["perovskite_1_3"]  # For AmbTemp and AmbHmd
            if "Irr" in env_features:
                env_modules.extend(["perovskite_1_1", "sanyo_5_1"])  # For Irr
            
            env_module_filter = " or ".join([f'r.Name == "{module}"' for module in env_modules])
            
            timezone_option = self._get_timezone_option()
            query2 = f'''
            {timezone_option}

            from(bucket:"{self.bucket}")
            {range_clause}
            |> filter(fn: (r) => r._measurement == "ParkData")
            |> filter(fn: (r) => {env_module_filter})
            |> filter(fn: (r) => {env_fields_filter})
            |> aggregateWindow(every: 5m, fn: mean, createEmpty: false)
            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            '''
            
            df2 = self.query_api.query_data_frame(query2)
            if isinstance(df2, list):
                df2 = pd.concat(df2, ignore_index=True)
        else:
            df2 = pd.DataFrame()

        # Combine dataframes
        if not df1.empty and not df2.empty:
            # Merge on time index
            df1.set_index('_time', inplace=True)
            df2.set_index('_time', inplace=True)
            df_combined = df1.join(df2, how='outer')
            df_combined.reset_index(inplace=True)
        elif not df1.empty:
            df_combined = df1
        elif not df2.empty:
            df_combined = df2
        else:
            df_combined = pd.DataFrame()

        # Add missing columns
        expected_columns = set(features + (outputs or []))
        for col in expected_columns:
            if col not in df_combined.columns:
                df_combined[col] = np.nan

        # Add the 'module_type' column
        if "Name" in df_combined.columns:
            df_combined["module_type"] = df_combined["Name"].apply(self.get_module_type)
        else:
            df_combined["module_type"] = "unknown"

        # Sort columns explicitly in desired order
        feature_order = ['_time', 'MAC', 'Name', 'module_type', 'Temp', 'I', 'U', 'P','AmbTemp', 'AmbHmd', 'Irr']
        df_combined = df_combined[[col for col in feature_order if col in df_combined.columns]]

        # Adjust timezone to German timezone
        df_combined = self._adjust_data_timezone(df_combined)

        # Post-process: Intelligent PV-specific imputation
        # df_combined = self._apply_pv_intelligent_imputation(df_combined)

        # Save to CSV
        if not df_combined.empty:
            df_combined.to_csv(f"training_data/{dataset_name}.csv", index=False)

        return df_combined
    
    def get_cnn_data(self, dataset_name: str, module_type: str, use_all_modules: bool, selected_modules: Union[str, list], features: list, outputs: list = None, date_selection: dict = None) -> pd.DataFrame:
        """
        Get data for CNN models (same as LSTM for now).
        """
        return self.get_lstm_data(dataset_name, module_type, use_all_modules, selected_modules, features, outputs, date_selection)
    
    def get_training_data(self, dataset_name: str, model_type: str, module_type: str, use_all_modules: str, selected_modules: Union[str, list], features: list, outputs: list = None, date_selection: dict = None) -> pd.DataFrame:
        """
        Unified method to get training data for any model type.
        Automatically uses chunked retrieval for large date ranges.
        
        Note: If outputs contains "P_normalized", it will be replaced with "P" for database query,
        since P_normalized is generated later by the DataValidator.
        """
        # Handle P_normalized in outputs - replace with P for database query
        if outputs is not None:
            # Create a copy to avoid modifying the original list
            outputs_for_query = outputs.copy()
            if "P_normalized" in outputs_for_query:
                outputs_for_query = [output if output != "P_normalized" else "P" for output in outputs_for_query]
                print(f"[INFO] Replaced 'P_normalized' with 'P' in outputs for database query: {outputs} -> {outputs_for_query}")
        else:
            outputs_for_query = outputs
        
        # Check if we need to use chunked retrieval for large date ranges
        if date_selection and isinstance(date_selection, dict):
            mode = date_selection.get("mode", "all")
            if mode == "custom":
                start_date = date_selection.get("start", "2024-08-16")
                end_date = date_selection.get("end", "2025-05-21")
                
                # Calculate date range in days
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
                days_diff = (end_dt - start_dt).days
                
                if days_diff > 30:  # If more than 30 days, use chunked approach
                    print(f"Large date range detected ({days_diff} days), using chunked retrieval...")
                    return self._get_training_data_chunked(
                        dataset_name, model_type, module_type, use_all_modules, 
                        selected_modules, features, outputs_for_query, start_date, end_date
                    )
        
        # Standard retrieval for smaller ranges
        if model_type.upper() == "CNN":
            return self.get_cnn_data(dataset_name, module_type, use_all_modules, selected_modules, features, outputs_for_query, date_selection)
        else:
            return self.get_lstm_data(dataset_name, module_type, use_all_modules, selected_modules, features, outputs_for_query, date_selection)

    def _get_training_data_chunked(self, dataset_name: str, model_type: str, module_type: str, use_all_modules: str, selected_modules: Union[str, list], features: list, outputs: list = None, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Retrieve training data in chunks to avoid timeout for large date ranges.
        
        Note: This method receives the already processed outputs (P_normalized -> P conversion done in get_training_data).
        """
        if self.client is None:
            # Generate sample data for offline mode
            return self._generate_sample_data(features, outputs, {"mode": "custom", "start": start_date, "end": end_date})
        
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Use 15-day chunks (reduced due to 5-minute aggregation)
        chunk_days = 15
        chunk_files = []
        
        current_start = start_dt
        chunk_count = 0
        
        # Create chunks directory
        chunks_dir = Path("training_data/chunks")
        chunks_dir.mkdir(parents=True, exist_ok=True)
        
        while current_start <= end_dt:
            chunk_end = min(current_start + pd.Timedelta(days=chunk_days - 1), end_dt)
            chunk_count += 1

            try:
                chunk_date_selection = {
                    "mode": "custom",
                    "start": current_start.strftime('%Y-%m-%d'),
                    "end": chunk_end.strftime('%Y-%m-%d')
                }
                
                # Use a unique dataset name for each chunk
                chunk_dataset_name = f"{dataset_name}_chunk_{chunk_count:03d}"
                
                # Get chunk data using the standard method
                if model_type.upper() == "CNN":
                    chunk_df = self.get_cnn_data(chunk_dataset_name, module_type, use_all_modules, selected_modules, features, outputs, chunk_date_selection)
                else:
                    chunk_df = self.get_lstm_data(chunk_dataset_name, module_type, use_all_modules, selected_modules, features, outputs, chunk_date_selection)
                
                if chunk_df is not None and not chunk_df.empty:
                    # Add missing columns
                    expected_columns = set(features + (outputs or []))
                    for col in expected_columns:
                        if col not in chunk_df.columns:
                            chunk_df[col] = np.nan
                    # The standard method automatically saves the file, so we need to find and move it
                    training_data_dir = Path("training_data")
                    auto_saved_files = list(training_data_dir.glob(f"*{chunk_dataset_name}*.csv"))
                    
                    if auto_saved_files:
                        # Use the automatically saved file
                        auto_file = auto_saved_files[0]
                        # Rename and move to chunks directory
                        chunk_file = chunks_dir / f"chunk_{chunk_count:03d}_{current_start.strftime('%Y%m%d')}_{chunk_end.strftime('%Y%m%d')}.csv"
                        auto_file.rename(chunk_file)
                        chunk_files.append(chunk_file)
                        print(f"  Chunk {chunk_count}: {len(chunk_df)} records -> {chunk_file.name}")
                    else:
                        # Fallback: save manually if auto-save didn't work
                        chunk_file = chunks_dir / f"chunk_{chunk_count:03d}_{current_start.strftime('%Y%m%d')}_{chunk_end.strftime('%Y%m%d')}.csv"
                        chunk_df.to_csv(chunk_file, index=False)
                        chunk_files.append(chunk_file)
                        print(f"  Chunk {chunk_count}: {len(chunk_df)} records -> {chunk_file.name}")
                else:
                    print(f"  Chunk {chunk_count}: No data")
                    
            except Exception as e:
                print(f"  Chunk {chunk_count} failed: {e}")
            
            # Start the next chunk the day after chunk_end
            current_start = chunk_end + pd.Timedelta(days=1)
        
        if chunk_files:
            # Combine all chunks
            print("Combining chunks...")
            combined_df = self._combine_chunks(chunk_files)
            print(f"Chunked retrieval complete: {len(combined_df)} total records")
            
            # Save final combined dataset
            final_file = Path("training_data") / f"{dataset_name}_combined_rawData.csv"
            combined_df.to_csv(final_file, index=False)
            print(f"Final dataset saved: {final_file}")
            
            return combined_df
        else:
            print("Chunked retrieval failed: No data retrieved")
            return pd.DataFrame()

    def _combine_chunks(self, chunk_files: list) -> pd.DataFrame:
        """
        Combine chunk files into a single DataFrame.
        """
        all_data = []
        
        for chunk_file in chunk_files:
            try:
                chunk_df = pd.read_csv(chunk_file)
                all_data.append(chunk_df)
                print(f"  Loaded {chunk_file.name}: {len(chunk_df)} records")
            except Exception as e:
                print(f"  Failed to load {chunk_file.name}: {e}")
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            return combined_df
        else:
            return pd.DataFrame()

    def _execute_query_with_timeout(self, query: str, timeout: int = 3600) -> pd.DataFrame:
        """Execute a query with extended timeout"""
        if self.client is None:
            return pd.DataFrame()
        
        try:
            # Execute query with extended timeout
            result = self.query_api.query_data_frame(query, timeout=timeout)
            
            if isinstance(result, list):
                return pd.concat(result, ignore_index=True)
            else:
                return result
                
        except Exception as e:
            print(f"Query execution failed: {e}")
            return pd.DataFrame()

    def _get_german_timezone_offset(self) -> str:
        """
        Get the current German timezone offset (1h for winter time, 2h for summer time).
        
        Returns:
            str: Timezone offset string for Flux query (e.g., "1h" or "2h")
        """
        # Get current time in German timezone
        german_tz = pytz.timezone('Europe/Berlin')
        now = datetime.now(german_tz)
        
        # Check if it's daylight saving time
        if now.dst():
            return "2h"  # Summer time (CEST)
        else:
            return "1h"  # Winter time (CET)
    
    def _get_timezone_option(self) -> str:
        """
        Get the timezone option string for Flux queries.
        
        Returns:
            str: Complete timezone option string for Flux
        """
        offset = self._get_german_timezone_offset()
        return f'import "timezone"\noption location = timezone.fixed(offset: {offset})'

    def _adjust_data_timezone(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adjust the timezone of the loaded data to German timezone.
        This ensures that even if the data comes in UTC, it's converted to the correct local time.
        
        Args:
            df (pd.DataFrame): DataFrame with _time column
            
        Returns:
            pd.DataFrame: DataFrame with adjusted timezone
        """
        if df.empty or "_time" not in df.columns:
            return df
        
        # Create a copy to avoid SettingWithCopyWarning
        df = df.copy()
        
        # Convert _time to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df["_time"]):
            df["_time"] = pd.to_datetime(df["_time"], utc=True)
        
        # Check if the timezone is already set
        if df["_time"].dt.tz is None:
            # Assume UTC and convert to German timezone
            german_tz = pytz.timezone('Europe/Berlin')
            df["_time"] = df["_time"].dt.tz_localize('UTC').dt.tz_convert(german_tz)
            print(f"OK Converted timezone from UTC to German timezone")
        elif df["_time"].dt.tz != pytz.timezone('Europe/Berlin'):
            # Convert from current timezone to German timezone
            german_tz = pytz.timezone('Europe/Berlin')
            df["_time"] = df["_time"].dt.tz_convert(german_tz)
            print(f"OK Converted timezone to German timezone")
        else:
            print(f"OK Data already in German timezone")
        
        return df

    def _apply_pv_intelligent_imputation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply intelligent PV-specific imputation to the DataFrame.
        Handles realistic PV patterns: gradual transitions, night values, etc.
        
        Args:
            df (pd.DataFrame): DataFrame with _time column
            
        Returns:
            pd.DataFrame: DataFrame with intelligently imputed values
        """
        if df.empty or '_time' not in df.columns:
            return df
        
        df = df.copy()
        df['_time'] = pd.to_datetime(df['_time'])
        
        # Define PV-specific columns
        power_columns = ['U', 'I', 'P']  # Electrical parameters
        sensor_columns = ['Temp', 'Irr']  # Sensor data
        env_columns = ['AmbTemp', 'AmbHmd']  # Environmental data
        
        # Get available columns
        available_power = [col for col in power_columns if col in df.columns]
        available_sensor = [col for col in sensor_columns if col in df.columns]
        available_env = [col for col in env_columns if col in df.columns]
        
        print(f"Applying intelligent PV imputation...")
        print(f"  Power columns: {available_power}")
        print(f"  Sensor columns: {available_sensor}")
        print(f"  Environmental columns: {available_env}")
        
        # Step 1: Handle night values (22:00-06:00)
        night_mask = (df['_time'].dt.hour >= 22) | (df['_time'].dt.hour <= 6)
        
        # Set power to 0 during night
        for col in available_power:
            df.loc[night_mask, col] = 0.0
            print(f"  Set night values (22:00-06:00) to 0 for {col}")
        
        # Step 2: Intelligent imputation for day values
        day_mask = ~night_mask
        
        for col in available_power + available_sensor:
            if col in df.columns:
                # Find abrupt transitions (sudden drops to 0)
                col_data = df.loc[day_mask, col].copy()
                
                # Identify abrupt drops (from > threshold to 0)
                threshold = 1.0  # Minimum meaningful value
                abrupt_drops = (col_data > threshold) & (col_data.shift(-1) == 0.0)
                
                if abrupt_drops.any():
                    print(f"  Found {abrupt_drops.sum()} abrupt drops in {col}")
                    
                    # Apply gradual transition for abrupt drops
                    for idx in col_data[abrupt_drops].index:
                        # Look ahead to find the end of the zero sequence
                        zero_sequence_end = idx
                        for i in range(1, 13):  # Max 1 hour ahead
                            if idx + i < len(col_data):
                                if col_data.iloc[idx + i] > threshold:
                                    zero_sequence_end = idx + i - 1
                                    break
                            else:
                                break
                        
                        # Apply gradual transition
                        if zero_sequence_end > idx:
                            start_value = col_data.iloc[idx]
                            end_value = col_data.iloc[zero_sequence_end + 1] if zero_sequence_end + 1 < len(col_data) else 0.0
                            
                            # Linear interpolation for the transition
                            for j in range(idx, zero_sequence_end + 1):
                                if j < len(col_data):
                                    progress = (j - idx) / (zero_sequence_end - idx + 1)
                                    interpolated_value = start_value * (1 - progress) + end_value * progress
                                    col_data.iloc[j] = interpolated_value
                    
                    # Update the DataFrame
                    df.loc[day_mask, col] = col_data
        
        # Step 3: Handle environmental data (different logic)
        for col in available_env:
            if col in df.columns:
                # For environmental data, use forward fill for short gaps
                df[col] = df[col].fillna(method='ffill', limit=12)  # 1 hour max
                df[col] = df[col].fillna(method='bfill', limit=12)
                df[col] = df[col].fillna(0.0)
        
        # Step 4: Final cleanup for any remaining NaN values
        for col in df.columns:
            if col not in ['_time', 'MAC', 'Name', 'module_type'] and df[col].dtype in ['float64', 'int64']:
                remaining_nan = df[col].isna().sum()
                if remaining_nan > 0:
                    print(f"  Filling {remaining_nan} remaining NaN values in {col}")
                    df[col] = df[col].fillna(0.0)
        
        print(f"OK Intelligent PV imputation complete")
        return df

    def get_forecast_preprocess_data(self, features: list, outputs: list = None, date_selection: dict = None, module_type: str = "Silicon", use_all_modules: str = "all", selected_modules: Union[str, list] = "all") -> pd.DataFrame:
        """
        Query and prepare input data for forecast preprocessing.
        Saves the result as results/forecast_data/rawData/latest_forecast_input_raw.csv (overwrites each time).
        Args:
            features (list): List of input feature names to retrieve.
            outputs (list): List of output feature names to retrieve.
            date_selection (dict): Date range selection.
            module_type (str): Module type filter (default: "Silicon").
            use_all_modules (str): Use all modules of this type (default: "all").
            selected_modules (Union[str, list]): Specific module names to filter (default: "all").
        Returns:
            pd.DataFrame: Combined DataFrame of requested fields.
        """
        if outputs is None:
            outputs = []

        if self.client is None:
            # Generate sample data for offline mode
            df_combined = self._generate_sample_data(features, outputs, date_selection)
        else:
            # Separate features into module-specific and environmental features
            module_features = [f for f in features if f not in ["AmbTemp", "AmbHmd", "Irr"]]
            env_features = [f for f in features if f in ["AmbTemp", "AmbHmd", "Irr"]]
            all_module_fields = module_features + outputs + env_features

            # Build date range clause with validation
            if isinstance(date_selection, dict):
                mode = date_selection.get("mode", "all").lower()
                if mode == "custom":
                    start = date_selection["start"]
                    end = date_selection["end"]
                    start, end = self._validate_date_range(start, end)
                    range_clause = f'|> range(start: time(v: "{start}T00:00:00Z"), stop: time(v: "{end}T23:59:59Z"))'
                elif mode == "last":
                    try:
                        hours = int(date_selection.get("value", 720))
                        range_clause = f'|> range(start: -{hours}h)'
                    except:
                        range_clause = f'|> range(start: time(v: "{self.MIN_DATA_DATE}T00:00:00Z"))'
                else:
                    range_clause = f'|> range(start: time(v: "{self.MIN_DATA_DATE}T00:00:00Z"))'
            else:
                range_clause = f'|> range(start: time(v: "{self.MIN_DATA_DATE}T00:00:00Z"))'

            # Query 1: Module-specific data (including outputs)
            if all_module_fields:
                module_fields_filter = " or ".join([f'r._field == "{field}"' for field in all_module_fields])
                if not use_all_modules and isinstance(selected_modules, list) and selected_modules:
                    module_filter = " or ".join([f'r.Name == "{module}"' for module in selected_modules])
                    timezone_option = self._get_timezone_option()
                    query1 = f'''
                    {timezone_option}

                    from(bucket:"{self.bucket}")
                    {range_clause}
                    |> filter(fn: (r) => r._measurement == "ParkData")
                    |> filter(fn: (r) => {module_filter})
                    |> filter(fn: (r) => {module_fields_filter})
                    |> aggregateWindow(every: 5m, fn: mean, createEmpty: true)
                    |> fill(value: 0.0)
                    |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
                    '''
                else:
                    timezone_option = self._get_timezone_option()
                    query1 = f'''
                    {timezone_option}

                    from(bucket:"{self.bucket}")
                    {range_clause}
                    |> filter(fn: (r) => r._measurement == "ParkData")
                    |> filter(fn: (r) => {module_fields_filter})
                    |> aggregateWindow(every: 5m, fn: mean, createEmpty: true)
                    |> fill(value: 0.0)
                    |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
                    '''
                df1 = self.query_api.query_data_frame(query1)
                if isinstance(df1, list):
                    df1 = pd.concat(df1, ignore_index=True)
            else:
                df1 = pd.DataFrame()

            # Query 2: Environmental data
            if env_features:
                env_fields_filter = " or ".join([f'r._field == "{field}"' for field in env_features])
                env_modules = ["perovskite_1_3"]
                if "Irr" in env_features:
                    env_modules.extend(["perovskite_1_1", "sanyo_5_1"])
                env_module_filter = " or ".join([f'r.Name == "{module}"' for module in env_modules])
                timezone_option = self._get_timezone_option()
                query2 = f'''
                {timezone_option}

                from(bucket:"{self.bucket}")
                {range_clause}
                |> filter(fn: (r) => r._measurement == "ParkData")
                |> filter(fn: (r) => {env_module_filter})
                |> filter(fn: (r) => {env_fields_filter})
                |> aggregateWindow(every: 5m, fn: mean, createEmpty: true)
                |> fill(value: 0.0)
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
                '''
                df2 = self.query_api.query_data_frame(query2)
                if isinstance(df2, list):
                    df2 = pd.concat(df2, ignore_index=True)
            else:
                df2 = pd.DataFrame()

            # Combine dataframes
            if not df1.empty and not df2.empty:
                df1.set_index('_time', inplace=True)
                df2.set_index('_time', inplace=True)
                df_combined = df1.join(df2, how='outer')
                df_combined.reset_index(inplace=True)
            elif not df1.empty:
                df_combined = df1
            elif not df2.empty:
                df_combined = df2
            else:
                df_combined = pd.DataFrame()

            # Add missing columns
            expected_columns = set(features + (outputs or []))
            for col in expected_columns:
                if col not in df_combined.columns:
                    df_combined[col] = np.nan

            # Add the 'module_type' column
            if "Name" in df_combined.columns:
                df_combined["module_type"] = df_combined["Name"].apply(self.get_module_type)
            else:
                df_combined["module_type"] = "unknown"

            # Sort columns explicitly in desired order
            feature_order = ['_time', 'MAC', 'Name', 'module_type', 'Temp', 'I', 'U', 'AmbTemp', 'AmbHmd', 'Irr']
            df_combined = df_combined[[col for col in feature_order if col in df_combined.columns]]

            # Adjust timezone to German timezone
            df_combined = self._adjust_data_timezone(df_combined)

            # Post-process: Intelligent PV-specific imputation
            #df_combined = self._apply_pv_intelligent_imputation(df_combined)

        # Save to CSV in forecast_data/rawData
        out_dir = Path("results/forecast_data/rawData")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "latest_forecast_input_raw.csv"
        df_combined.to_csv(out_path, index=False)
        print(f"OK Forecast input data saved: {out_path}")
        return df_combined



if __name__ == "__main__":
    influx_client = influxClient("influxData_api/data/.env")
    # df = influx_client.query_parkdata_last_xh(20)
    # df = influx_client.query_parkdata_last_xh(48)
    df = influx_client.get_lstm_data(
        dataset_name="example_lstm_dataset",
        module_type="both",
        use_all_modules="Use all modules of this type",
        selected_modules=[],
        features=["Temp", "AmbTemp", "AmbHmd", "Irr", "Datetime"],
        outputs=["Temp"],
        date_selection= {
            "mode": "custom",
            "start": "2025-05-20",
            "end": "2025-05-24"
        }
    )
    print(df.head())