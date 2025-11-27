#!/usr/bin/env python3
"""
Offline Test for InfluxDB Data Query
Tests data query functionality without Docker containers
"""

import sys
import os
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import socket
import urllib.request
import urllib.error
from typing import Optional

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

# Add parent directories to path to import modules
sys.path.append(str(Path(__file__).parent.parent / "influxData_api" / "data"))

# Import the original influxDB_client (now supports both modes)
from influxDB_client import influxClient

class OfflineInfluxDBTest:
    def __init__(self, config_file: str = "test_lstm_model_settings.json"):
        """
        Initializes the offline test with configuration
        
        Args:
            config_file: Path to JSON configuration file
        """
        self.config_file = config_file
        self.config = self.load_config()
        
        # Determine date range and configName for folder structure
        date_sel = self.config.get('date_selection', {})
        start = date_sel.get('start', 'unknownStart').replace('-', '')
        end = date_sel.get('end', 'unknownEnd').replace('-', '')
        config_name = self.config.get('model_name', 'unknownConfig')
        self.rawdata_dir = Path(__file__).parent.parent / "results" / "training_data" / "rawData" / f"{start}_{end}"
        self.chunks_dir = self.rawdata_dir / "chunks"
        self.model_configs_dir = Path(__file__).parent.parent / "results" / "model_configs"
        self.rawdata_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.model_configs_dir.mkdir(parents=True, exist_ok=True)

        print(f"Raw data will be saved in: {self.rawdata_dir}")
        print(f"Chunks will be saved in: {self.chunks_dir}")
        print(f"Model configurations will be saved in: {self.model_configs_dir}")
        
        # Initialize client with extended timeout (60 minutes)
        self.client = influxClient(env_path=str(Path(__file__).parent / ".env"))
        if hasattr(self.client, 'query_api') and hasattr(self.client.query_api, '_api_client'):
            # Set extended timeout for large queries
            self.client.query_api._api_client.timeout = 3600  # 60 minutes
        
        self.splitter = self.client.splitter if hasattr(self.client, 'splitter') else None
        
    def load_config(self) -> dict:
        """Loads test configuration from JSON file"""
        # Check if config_file is a full path or just filename
        if Path(self.config_file).is_absolute() or '/' in self.config_file or '\\' in self.config_file:
            # Full path provided
            config_path = Path(self.config_file)
        else:
            # Just filename provided, look in results/model_configs/
            config_path = Path(__file__).parent.parent / "results" / "model_configs" / self.config_file
        
        if not config_path.exists():
            # Create default configuration
            default_config = {
                "model_name": "test_model_offline",
                "model_type": "LSTM",
                "module_type": "Silicon",
                "use_all_modules": "Use all modules of this type",
                "selected_modules": "all",
                "features": ["U", "I", "P", "T", "G"],
                "output": ["Current", "Voltage", "Power"],
                "date_selection": {
                    "mode": "custom",
                    "start": "2024-08-17",
                    "end": "2025-06-22"
                },
                "influxdb_config": {
                    "host": "localhost",
                    "port": 8086,
                    "database": "pv_data",
                    "username": "admin",
                    "password": "admin123"
                }
            }
            
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=4)
            
            print(f"Default configuration created: {config_path}")
            return default_config
        
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def test_network_connectivity(self) -> bool:
        """Tests basic network connectivity to InfluxDB server"""
        print("Testing network connectivity...")
        
        try:
            # Extract host and port from URL
            url = os.getenv("INFLUX_URL", "http://193.196.55.253:8086")
            if url.startswith("http://"):
                host = url[7:]  # Remove "http://"
            elif url.startswith("https://"):
                host = url[8:]  # Remove "https://"
            else:
                host = url
            
            if ":" in host:
                host, port_str = host.split(":")
                port = int(port_str)
            else:
                port = 8086
            
            # Test TCP connection
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10)  # 10 second timeout
            result = sock.connect_ex((host, port))
            sock.close()
            
            if result == 0:
                print("OK Network connectivity: OK")
                return True
            else:
                print(f"FAIL Network connectivity failed")
                return False
                
        except Exception as e:
            print(f"FAIL Network connectivity failed: {e}")
            return False
    
    def test_http_ping(self) -> bool:
        """Tests HTTP ping to InfluxDB API"""
        print("Testing HTTP ping...")
        
        try:
            url = os.getenv("INFLUX_URL", "http://193.196.55.253:8086")
            ping_url = f"{url}/ping"
            
            # Try to ping the InfluxDB API with proper headers
            req = urllib.request.Request(ping_url)
            req.add_header('User-Agent', 'InfluxDB-Test/1.0')
            req.add_header('Accept', 'application/json')
            
            # Try without authentication first
            with urllib.request.urlopen(req, timeout=10) as response:
                status = response.getcode()
                
                if status == 204:  # InfluxDB ping returns 204 No Content
                    print("OK HTTP ping: OK")
                    return True
                elif status == 401:  # Unauthorized - API is working but needs auth
                    print("OK HTTP ping: OK (requires auth)")
                    return True
                else:
                    print(f"WARN HTTP ping: Unexpected status {status}")
                    return False
                    
        except urllib.error.HTTPError as e:
            if e.code == 401:
                print("OK HTTP ping: OK (requires auth)")
                return True
            print(f"FAIL HTTP ping failed: {e.code}")
            return False
        except urllib.error.URLError as e:
            print(f"FAIL HTTP ping failed: {e.reason}")
            return False
        except Exception as e:
            print(f"FAIL HTTP ping failed: {e}")
            return False
    
    def test_authentication_scenarios(self) -> bool:
        """Tests different authentication scenarios"""
        print("Testing authentication...")
        
        try:
            # Test 1: List buckets (requires read permissions)
            buckets_query = '''
            buckets()
            '''
            buckets_result = self.client.query_api.query_data_frame(buckets_query)
            print(f"OK Authentication: OK ({len(buckets_result)} buckets available)")
            return True
            
        except Exception as e:
            print(f"FAIL Authentication failed: {e}")
            return False
    
    def test_connection(self) -> bool:
        """Tests basic connection to InfluxDB"""
        print("Testing connection...")
        
        try:
            # Test basic connection
            if self.client is None or self.client.query_api is None:
                print("FAIL Connection failed: Client not initialized")
                return False
            
            # Try a simple query to test connection
            test_query = f'''
            from(bucket:"{self.client.bucket}")
            |> range(start: -1h)
            |> limit(n: 1)
            '''
            
            result = self.client.query_api.query_data_frame(test_query)
            print("OK Connection: OK")
            return True
            
        except Exception as e:
            print(f"FAIL Connection failed: {e}")
            return False
    
    def test_data_retrieval(self) -> Optional[pd.DataFrame]:
        """Tests data retrieval from InfluxDB"""
        print("Retrieving data (this may take several minutes)...")
        
        try:
            print("  Starting query...")
            df = self.client.get_training_data(
                dataset_name=self.config['model_name'],
                model_type=self.config['model_type'],
                module_type=self.config['module_type'],
                use_all_modules=self.config['use_all_modules'],
                selected_modules=self.config['selected_modules'],
                features=self.config['features'],
                outputs=self.config['output'],
                date_selection=self.config['date_selection']
            )
            
            if df is not None and not df.empty:
                print(f"OK Data retrieval: OK ({len(df)} records)")
                
                # Save results
                self.save_results(df)
                return df
            else:
                print("FAIL Data retrieval failed: No data")
                return None
                
        except Exception as e:
            if "timeout" in str(e).lower():
                print("FAIL Data retrieval failed: Query timeout (try reducing date range)")
            else:
                print(f"FAIL Data retrieval failed: {e}")
            return None
    
    def test_data_splitting(self, df: pd.DataFrame) -> Optional[bool]:
        """Tests data splitting for training/validation"""
        print("Splitting data...")
        
        try:
            # Configure splitting
            split_config = {
                "use_validation_set": "Yes",
                "validation_split": 0.15
            }
            
            # Use the client's built-in splitter if available, otherwise create a simple one
            if hasattr(self.client, 'splitter') and self.client.splitter is not None:
                train_data, val_data = self.client.splitter.split_data(
                    df, 
                    validation_split=split_config["validation_split"],
                    shuffle=True
                )
            else:
                # Simple fallback splitter
                if split_config["validation_split"] > 0:
                    n = round(1 / split_config["validation_split"])
                else:
                    n = len(df) + 1
                
                idx = range(len(df))
                val_mask = [(i % n) == 0 for i in idx]
                train_mask = [not mask for mask in val_mask]
                
                train_data = df[train_mask]
                val_data = df[val_mask]
            
            print(f"OK Data splitting: OK (train: {len(train_data)}, val: {len(val_data)})")
            
            # Save split data
            if not (isinstance(train_data, pd.DataFrame) and isinstance(val_data, pd.DataFrame)):
                print("FAIL Split result is not a DataFrame!")
                return False
            self.save_split_results(train_data, val_data)
            return True
            
        except Exception as e:
            print(f"FAIL Data splitting failed: {e}")
            return False
    
    def save_results(self, df: pd.DataFrame):
        """Saves raw data and statistics in the new results/ structure"""
        date_sel = self.config.get('date_selection', {})
        start = date_sel.get('start', 'unknownStart').replace('-', '')
        end = date_sel.get('end', 'unknownEnd').replace('-', '')
        config_name = self.config.get('model_name', 'unknownConfig')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Save raw data
        csv_path = self.rawdata_dir / f"{start}_{end}_{config_name}_raw.csv"
        df.to_csv(csv_path, index=False)
        # Save statistics
        stats = {
            "model_name": config_name,
            "timestamp": timestamp,
            "record_count": len(df),
            "columns": list(df.columns),
            "date_range": {
                "start": str(df.index.min()) if hasattr(df.index, 'min') else "N/A",
                "end": str(df.index.max()) if hasattr(df.index, 'max') else "N/A"
            },
            "client_mode": getattr(self.client, 'mode', 'unknown'),
            "config": self.config
        }
        stats_path = self.rawdata_dir / f"{start}_{end}_{config_name}_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=4)
        print(f"Results saved: {csv_path}")
    
    def save_split_results(self, train_data: pd.DataFrame, val_data: pd.DataFrame):
        """Saves training and validation data in the new results/ structure"""
        date_sel = self.config.get('date_selection', {})
        start = date_sel.get('start', 'unknownStart').replace('-', '')
        end = date_sel.get('end', 'unknownEnd').replace('-', '')
        config_name = self.config.get('model_name', 'unknownConfig')
        # Save training data
        train_path = self.rawdata_dir / f"{start}_{end}_{config_name}_train.csv"
        train_data.to_csv(train_path, index=False)
        # Save validation data
        val_path = self.rawdata_dir / f"{start}_{end}_{config_name}_val.csv"
        val_data.to_csv(val_path, index=False)
        print(f"Split data saved: {train_path}, {val_path}")
    
    def run_full_test(self) -> bool:
        """Runs the complete test"""
        print("=" * 50)
        print(f"INFLUXDB DATA QUERY TEST - Model: {self.config.get('model_name', 'unknown')}")
        print("=" * 50)
        
        # Test 1: Network connectivity
        success = self.test_network_connectivity()
        if not success:
            return False
        
        # Test 2: HTTP ping
        success = self.test_http_ping()
        if not success:
            print("WARN HTTP ping failed, continuing...")
        
        # Test 3: Authentication scenarios
        success = self.test_authentication_scenarios()
        if not success:
            return False
        
        # Test 4: Connection
        success = self.test_connection()
        if not success:
            return False
        
        # Test 5: Data retrieval
        df = self.test_data_retrieval()
        if df is None:
            return False
        
        # Test 6: Data splitting (use data from test_data_retrieval)
        success = self.test_data_splitting(df)
        if not success:
            return False
        
        print("\n" + "=" * 50)
        print("OK ALL TESTS PASSED")
        print("=" * 50)
        return True

def main():
    """Main function"""
    test = OfflineInfluxDBTest()
    
    try:
        success = test.run_full_test()
        if success:
            print("SUCCESS: InfluxDB test completed successfully")
            sys.exit(0)
        else:
            print("ERROR: InfluxDB test failed")
            sys.exit(1)
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 