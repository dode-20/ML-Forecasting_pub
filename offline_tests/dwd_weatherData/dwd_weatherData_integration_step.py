import requests
import pandas as pd
from datetime import datetime, timedelta
import zipfile
import io
import os
import glob
import numpy as np
import pygrib
import tempfile
import re
import pytz

# Optional imports for visualization
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

class DWDOptimizedIntegrator:
    """Optimized DWD weather data integrator with solar data support"""
    
    def __init__(self, station_id: str = "04931"):  # Stuttgart-Echterdingen for weather data
        self.station_id = station_id  # 04931 for temperature, humidity, cloudiness, precipitation
        self.solar_station_id = "04928"  # Stuttgart Schnarrenberg for solar data only
        self.base_url = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate"
        
        # German timezone for UTC to local time conversion
        self.german_tz = pytz.timezone('Europe/Berlin')
        
        # Create necessary directories
        os.makedirs("results/weather_data/rawData", exist_ok=True)
        os.makedirs("results/weather_data/cleanData", exist_ok=True)
        os.makedirs("results/weather_data/forecast_data/rawData", exist_ok=True)
        os.makedirs("results/weather_data/visualizations", exist_ok=True)
    
    def convert_utc_to_german_time(self, utc_timestamp):
        """
        Convert UTC timestamp to German local time (CET/CEST) with automatic DST handling.
        
        Args:
            utc_timestamp: UTC timestamp (timezone-naive, assumed to be UTC)
            
        Returns:
            German local timestamp (timezone-naive)
        """
        # Ensure timestamp is timezone-aware UTC
        if utc_timestamp.tz is None:
            utc_timestamp = pytz.utc.localize(utc_timestamp)
        
        # Convert to German timezone
        german_time = utc_timestamp.astimezone(self.german_tz)
        
        # Return timezone-naive timestamp (as string for CSV compatibility)
        return german_time.strftime('%Y-%m-%d %H:%M:%S')
    
    def convert_utc_timestamps_in_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert all UTC timestamps in a DataFrame to German local time.
        
        Args:
            df: DataFrame with 'timestamp' column containing UTC timestamps
            
        Returns:
            DataFrame with converted German local timestamps
        """
        if 'timestamp' not in df.columns:
            print("[WARN] No timestamp column found for UTC conversion")
            return df
        
        print(f"[INFO] Converting {len(df)} UTC timestamps to German local time...")
        
        # Convert timestamps
        df_converted = df.copy()
        df_converted['timestamp'] = df_converted['timestamp'].apply(self.convert_utc_to_german_time)
        
        # Show conversion example
        if len(df_converted) > 0:
            print(f"[INFO] Example conversion:")
            print(f"[INFO]   Original UTC: {df['timestamp'].iloc[0]}")
            print(f"[INFO]   German time:  {df_converted['timestamp'].iloc[0]}")
        
        return df_converted

    def add_time_features(self, data: pd.DataFrame, time_features: list) -> pd.DataFrame:
        """
        Adds the desired time features as columns (e.g. hour, minute, weekday, month, day_of_year).
        Works with German local time timestamps (CET/CEST).
        
        Args:
            data (pd.DataFrame): DataFrame with timestamp column (German local time)
            time_features (list): List of the desired time features
        Returns:
            pd.DataFrame: DataFrame with new time feature columns
        """
        if 'timestamp' not in data.columns:
            raise ValueError("timestamp column is required to extract time features.")
        data = data.copy()
        
        # Convert timestamp to datetime if it's a string
        if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
            data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Ensure timestamps are timezone-naive (German local time)
        if data['timestamp'].dt.tz is not None:
            data['timestamp'] = data['timestamp'].dt.tz_localize(None)
        
        # Extract time features from German local time
        if 'hour' in time_features:
            data['hour'] = data['timestamp'].dt.hour
        if 'minute' in time_features:
            data['minute'] = data['timestamp'].dt.minute
        if 'weekday' in time_features:
            data['weekday'] = data['timestamp'].dt.weekday
        if 'month' in time_features:
            data['month'] = data['timestamp'].dt.month
        if 'day_of_year' in time_features:
            data['day_of_year'] = data['timestamp'].dt.dayofyear
        
        print(f"[INFO] Extracted time features from German local time: {time_features}")
        return data

    def get_historical_temperature_humidity(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Download and extract temperature TXT files"""
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        current_dt = start_dt.replace(day=1)
        
        while current_dt <= end_dt:
            year = current_dt.year
            month = current_dt.month
            filename = f"stundenwerte_TU_{self.station_id}_akt.zip"
            url = f"{self.base_url}/hourly/air_temperature/recent/{filename}"
            
            try:
                response = requests.get(url)
                response.raise_for_status()
                
                # Extract ZIP and save TXT files
                with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
                    for txt_file in zip_file.namelist():
                        if txt_file.endswith('.txt'):
                            # Extract TXT file content
                            txt_content = zip_file.read(txt_file)
                            
                            # Save TXT file directly
                            txt_filename = txt_file.split('/')[-1]  # Get just filename
                            save_path = f"results/weather_data/rawData/airTemperature/{txt_filename}"
                            os.makedirs(os.path.dirname(save_path), exist_ok=True)
                            
                            with open(save_path, 'wb') as f:
                                f.write(txt_content)
                            print(f"Extracted: {txt_filename}")
                
            except Exception as e:
                print(f"Failed: {filename} - {e}")
            
            if current_dt.month == 12:
                current_dt = current_dt.replace(year=current_dt.year + 1, month=1)
            else:
                current_dt = current_dt.replace(month=current_dt.month + 1)
        
        return pd.DataFrame()
    
    def get_historical_cloudiness(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Download and extract cloudiness TXT files"""
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        current_dt = start_dt.replace(day=1)
        
        while current_dt <= end_dt:
            year = current_dt.year
            month = current_dt.month
            filename = f"stundenwerte_N_{self.station_id}_akt.zip"
            url = f"{self.base_url}/hourly/cloudiness/recent/{filename}"
            
            try:
                response = requests.get(url)
                response.raise_for_status()
                
                # Extract ZIP and save TXT files
                with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
                    for txt_file in zip_file.namelist():
                        if txt_file.endswith('.txt'):
                            # Extract TXT file content
                            txt_content = zip_file.read(txt_file)
                            
                            # Save TXT file directly
                            txt_filename = txt_file.split('/')[-1]  # Get just filename
                            save_path = f"results/weather_data/rawData/cloudiness/{txt_filename}"
                            os.makedirs(os.path.dirname(save_path), exist_ok=True)
                            
                            with open(save_path, 'wb') as f:
                                f.write(txt_content)
                            print(f"Extracted: {txt_filename}")
                
            except Exception as e:
                print(f"Failed: {filename} - {e}")
            
            if current_dt.month == 12:
                current_dt = current_dt.replace(year=current_dt.year + 1, month=1)
            else:
                current_dt = current_dt.replace(month=current_dt.month + 1)
        
        return pd.DataFrame()
    
    def get_historical_precipitation(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Download and extract precipitation TXT files"""
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        current_dt = start_dt.replace(day=1)
        
        while current_dt <= end_dt:
            year = current_dt.year
            month = current_dt.month
            filename = f"stundenwerte_RR_{self.station_id}_akt.zip"
            url = f"{self.base_url}/hourly/precipitation/recent/{filename}"
            
            try:
                response = requests.get(url)
                response.raise_for_status()
                
                # Extract ZIP and save TXT files
                with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
                    for txt_file in zip_file.namelist():
                        if txt_file.endswith('.txt'):
                            # Extract TXT file content
                            txt_content = zip_file.read(txt_file)
                            
                            # Save TXT file directly
                            txt_filename = txt_file.split('/')[-1]  # Get just filename
                            save_path = f"results/weather_data/rawData/precipitation/{txt_filename}"
                            os.makedirs(os.path.dirname(save_path), exist_ok=True)
                            
                            with open(save_path, 'wb') as f:
                                f.write(txt_content)
                            print(f"Extracted: {txt_filename}")
                
            except Exception as e:
                print(f"Failed: {filename} - {e}")
            
            if current_dt.month == 12:
                current_dt = current_dt.replace(year=current_dt.year + 1, month=1)
            else:
                current_dt = current_dt.replace(month=current_dt.month + 1)
        
        return pd.DataFrame()

    def get_historical_solar_data(self, start_date: str, end_date: str):
        """Download historical solar data from DWD"""
        print("="*60)
        print("DOWNLOADING HISTORICAL SOLAR DATA")
        print("="*60)
        
        # Create solar data directory
        os.makedirs("results/weather_data/rawData/solar", exist_ok=True)
        
        # Use sun data from station 04928 (Stuttgart Schnarrenberg) - more recent data available
        solar_filename = f"stundenwerte_SD_{self.solar_station_id}_akt.zip"
        solar_url = f"https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/sun/recent/{solar_filename}"
        
        print(f"Downloading solar data: {solar_filename}")
        print(f"URL: {solar_url}")
        
        try:
            response = requests.get(solar_url, timeout=30)
            response.raise_for_status()
            
            # Save the ZIP file
            solar_zip_path = f"results/weather_data/rawData/solar/{solar_filename}"
            with open(solar_zip_path, 'wb') as f:
                f.write(response.content)
            
            print(f"Downloaded: {solar_zip_path}")
            
            # Extract the ZIP file
            with zipfile.ZipFile(solar_zip_path, 'r') as zip_ref:
                zip_ref.extractall("results/weather_data/rawData/solar/")
            
            print("Extracted solar data files")
            
        except requests.exceptions.RequestException as e:
            print(f"Error downloading solar data: {e}")
        except Exception as e:
            print(f"Error processing solar data: {e}")

    def get_10min_temperature_humidity(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Download 10-minute temperature and humidity data from DWD"""
        print("="*60)
        print("DOWNLOADING 10-MINUTE TEMPERATURE AND HUMIDITY DATA")
        print("="*60)
        
        # Create directory
        os.makedirs("results/weather_data/rawData/10min_airTemperature", exist_ok=True)
        
        # Download 10-minute data
        filename = f"10minutenwerte_TU_{self.station_id}_akt.zip"
        url = f"{self.base_url}/10_minutes/air_temperature/recent/{filename}"
        
        print(f"Downloading 10-minute temperature/humidity data: {filename}")
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Save ZIP file
            zip_path = f"results/weather_data/rawData/10min_airTemperature/{filename}"
            with open(zip_path, 'wb') as f:
                f.write(response.content)
            
            # Extract ZIP file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall("results/weather_data/rawData/10min_airTemperature/")
            
            print(f"Downloaded and extracted: {filename}")
            
        except requests.exceptions.RequestException as e:
            print(f"Error downloading 10-minute temperature/humidity data: {e}")
        except Exception as e:
            print(f"Error processing 10-minute temperature/humidity data: {e}")

    def get_10min_precipitation(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Download 10-minute precipitation data from DWD"""
        print("="*60)
        print("DOWNLOADING 10-MINUTE PRECIPITATION DATA")
        print("="*60)
        
        # Create directory
        os.makedirs("results/weather_data/rawData/10min_precipitation", exist_ok=True)
        
        # Download 10-minute data
        filename = f"10minutenwerte_nieder_{self.station_id}_akt.zip"
        url = f"{self.base_url}/10_minutes/precipitation/recent/{filename}"
        
        print(f"Downloading 10-minute precipitation data: {filename}")
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Save ZIP file
            zip_path = f"results/weather_data/rawData/10min_precipitation/{filename}"
            with open(zip_path, 'wb') as f:
                f.write(response.content)
            
            # Extract ZIP file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall("results/weather_data/rawData/10min_precipitation/")
            
            print(f"Downloaded and extracted: {filename}")
            
        except requests.exceptions.RequestException as e:
            print(f"Error downloading 10-minute precipitation data: {e}")
        except Exception as e:
            print(f"Error processing 10-minute precipitation data: {e}")

    def get_10min_solar_data(self, start_date: str, end_date: str):
        """Download 10-minute solar data from DWD"""
        print("="*60)
        print("DOWNLOADING 10-MINUTE SOLAR DATA")
        print("="*60)
        
        # Create solar data directory
        os.makedirs("results/weather_data/rawData/10min_solar", exist_ok=True)
        
        # Download 10-minute solar data
        filename = f"10minutenwerte_SOLAR_{self.solar_station_id}_akt.zip"
        url = f"{self.base_url}/10_minutes/solar/recent/{filename}"
        
        print(f"Downloading 10-minute solar data: {filename}")
        print(f"URL: {url}")
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Save the ZIP file
            solar_zip_path = f"results/weather_data/rawData/10min_solar/{filename}"
            with open(solar_zip_path, 'wb') as f:
                f.write(response.content)
            
            print(f"Downloaded: {solar_zip_path}")
            
            # Extract the ZIP file
            with zipfile.ZipFile(solar_zip_path, 'r') as zip_ref:
                zip_ref.extractall("results/weather_data/rawData/10min_solar/")
            
            print("Extracted 10-minute solar data files")
            
        except requests.exceptions.RequestException as e:
            print(f"Error downloading 10-minute solar data: {e}")
        except Exception as e:
            print(f"Error processing 10-minute solar data: {e}")

    def validate_and_interpolate_weather_data(self, df: pd.DataFrame, data_type: str = "historical") -> pd.DataFrame:
        """
        Validate weather data and interpolate error values and missing data
        
        Args:
            df: DataFrame with weather data
            data_type: Type of data ("historical" or "forecast")
            
        Returns:
            pd.DataFrame: Cleaned and interpolated DataFrame
        """
        print("="*60)
        print(f"VALIDATING {data_type.upper()} WEATHER DATA")
        print("="*60)
        
        # Define error values for each parameter
        base_error_values = {
            'TT_10': -999,      # 10-minute temperature
            'RF_10': -999,      # 10-minute humidity
            'V_N': -1,          # Cloudiness (still hourly)
            'RWS_10': -999,     # 10-minute precipitation
            'RWS_IND_10': -999, # 10-minute precipitation indicator
            'SD_10': -999,      # 10-minute sunshine duration
            'GS_10': -999       # 10-minute global solar radiation
        }

            
        error_values = base_error_values
        
        # Create a copy to avoid modifying original
        df_clean = df.copy()
        
        total_errors = 0
        total_missing = 0
        
        #1. First, replace ALL error values in all relevant columns with NaN.
        for column, error_value in error_values.items():
            if column not in df_clean.columns:
                print(f"WARNING: Column {column} not found in data")
                continue
            error_mask = df_clean[column] == error_value
            error_count = error_mask.sum()
            total_errors += error_count
            if error_count > 0:
                df_clean.loc[error_mask, column] = np.nan
                print(f"{column}: Replaced {error_count} error values ({error_value}) with NaN")
        
        # 2. Count missing values (after all error values have been removed)
        for column in error_values.keys():
            if column in df_clean.columns:
                missing_count = df_clean[column].isna().sum()
                total_missing += missing_count
                print(f"{column}: Missing values (NaN) after error removal: {missing_count}")
        
        # 3. Now perform interpolation/filling - ONLY for missing values (NaN)
        for column in error_values.keys():
            if column not in df_clean.columns:
                continue
            
            # Check if there are any missing values (NaN) in this column
            missing_count = df_clean[column].isna().sum()
            
            if missing_count > 0:
                if column in ['TT_10', 'RF_10', 'RWS_10', 'GS_10']:
                    # Continuous variables: linear interpolation ONLY for missing values
                    df_clean[column] = df_clean[column].interpolate(method='linear', limit_direction='both')
                    print(f"{column}: Applied linear interpolation for {missing_count} missing values")
                elif column in ['V_N', 'RWS_IND_10', 'SD_10']:
                    # Discrete/categorical variables: forward fill then backward fill ONLY for missing values
                    df_clean[column] = df_clean[column].fillna(method='ffill').fillna(method='bfill')
                    print(f"{column}: Applied forward/backward fill for {missing_count} missing values")
                
                # Fallback for remaining gaps
                remaining_missing = df_clean[column].isna().sum()
                if remaining_missing > 0:
                    print(f"{column}: WARNING: {remaining_missing} values still missing after interpolation")
                    if column == 'TT_10':
                        df_clean[column].fillna(15.0, inplace=True)
                    elif column == 'RF_10':
                        df_clean[column].fillna(60.0, inplace=True)
                    elif column == 'V_N':
                        df_clean[column].fillna(4, inplace=True)
                    elif column == 'RWS_10':
                        df_clean[column].fillna(0.0, inplace=True)
                    elif column == 'RWS_IND_10':
                        df_clean[column].fillna(0, inplace=True)
                    elif column == 'SD_10':
                        df_clean[column].fillna(0.0, inplace=True)  # Keine Sonnenscheindauer
                    elif column == 'GS_10':
                        df_clean[column].fillna(0.0, inplace=True)  # Keine Globalstrahlung
                    print(f"{column}: Filled remaining gaps with default values")
            else:
                print(f"{column}: No missing values - keeping original data unchanged")
        
        print(f"\nValidation summary:")
        print(f"   Total error values replaced: {total_errors}")
        print(f"   Total missing values interpolated: {total_missing}")
        print(f"   Data quality improved: {total_errors + total_missing} values corrected")
        
        # Final data quality check
        print(f"\nFinal data quality:")
        for column in error_values.keys():
            if column in df_clean.columns:
                missing = df_clean[column].isna().sum()
                total = len(df_clean)
                percentage = (missing / total) * 100
                print(f"   {column}: {missing} missing ({percentage:.1f}%)")
        
        return df_clean

    def process_txt_files_to_csv(self):
        """Process all downloaded TXT files and create combined CSV"""
        print("="*60)
        print("PROCESSING TXT FILES TO CSV")
        print("="*60)
        
        # Initialize combined DataFrame
        combined_data = []
        
        # 1. Process 10-minute temperature/humidity files (TU) from station 04931
        print("\n1. Processing 10-minute temperature/humidity files from station 04931...")
        tu_files = glob.glob("results/weather_data/rawData/10min_airTemperature/produkt_zehn_min_tu_*.txt")
        
        for tu_file in tu_files:
            try:
                # Read TU file with semicolon separator and German encoding
                df_tu = pd.read_csv(tu_file, sep=';', encoding='latin1', header=0)
                
                print(f"  TU columns: {list(df_tu.columns)}")
                
                # Extract timestamp
                if 'MESS_DATUM' in df_tu.columns:
                    df_tu['timestamp'] = pd.to_datetime(df_tu['MESS_DATUM'], format='%Y%m%d%H%M')
                
                # Extract TT_10, RF_10
                for idx, row in df_tu.iterrows():
                    timestamp = df_tu.loc[idx, 'timestamp']
                    
                    data_row = {'timestamp': timestamp}
                    
                    # Extract temperature (TT_10)
                    if 'TT_10' in df_tu.columns:
                        data_row['TT_10'] = df_tu.loc[idx, 'TT_10']
                    
                    # Extract humidity (RF_10)
                    if 'RF_10' in df_tu.columns:
                        data_row['RF_10'] = df_tu.loc[idx, 'RF_10']
                    
                    combined_data.append(data_row)
                
                print(f"  Processed: {os.path.basename(tu_file)}")
                
            except Exception as e:
                print(f"  Error processing {tu_file}: {e}")
        
        # 2. Process hourly cloudiness files (N) from station 04931 (still hourly)
        print("\n2. Processing hourly cloudiness files from station 04931...")
        n_files = glob.glob("results/weather_data/rawData/cloudiness/produkt_n_stunde_*.txt")
        
        for n_file in n_files:
            try:
                # Read N file
                df_n = pd.read_csv(n_file, sep=';', encoding='latin1', header=0)
                
                print(f"  N columns: {list(df_n.columns)}")
                
                # Extract timestamp
                if 'MESS_DATUM' in df_n.columns:
                    df_n['timestamp'] = pd.to_datetime(df_n['MESS_DATUM'], format='%Y%m%d%H')
                
                # Extract V_N (cloudiness)
                for idx, row in df_n.iterrows():
                    timestamp = df_n.loc[idx, 'timestamp']
                    
                    # Find existing row with same timestamp or create new one
                    existing_row = None
                    for data_row in combined_data:
                        if data_row['timestamp'] == timestamp:
                            existing_row = data_row
                            break
                    
                    if existing_row is None:
                        existing_row = {'timestamp': timestamp}
                        combined_data.append(existing_row)
                    
                    if ' V_N' in df_n.columns:
                        existing_row['V_N'] = df_n.loc[idx, ' V_N']
                    
                print(f"  Processed: {os.path.basename(n_file)}")
                
            except Exception as e:
                print(f"  Error processing {n_file}: {e}")
        
        # 3. Process 10-minute precipitation files from station 04931
        print("\n3. Processing 10-minute precipitation files from station 04931...")
        precip_files = glob.glob("results/weather_data/rawData/10min_precipitation/produkt_zehn_min_rr_*.txt")
        
        for precip_file in precip_files:
            try:
                # Read precipitation file
                df_precip = pd.read_csv(precip_file, sep=';', encoding='latin1', header=0)
                
                print(f"  Precipitation columns: {list(df_precip.columns)}")
                
                # Extract timestamp
                if 'MESS_DATUM' in df_precip.columns:
                    df_precip['timestamp'] = pd.to_datetime(df_precip['MESS_DATUM'], format='%Y%m%d%H%M')
                
                # Extract RWS_10, RWS_IND_10
                for idx, row in df_precip.iterrows():
                    timestamp = df_precip.loc[idx, 'timestamp']
                    
                    # Find existing row with same timestamp or create new one
                    existing_row = None
                    for data_row in combined_data:
                        if data_row['timestamp'] == timestamp:
                            existing_row = data_row
                            break
                    
                    if existing_row is None:
                        existing_row = {'timestamp': timestamp}
                        combined_data.append(existing_row)
                    
                    # Extract precipitation parameters
                    if 'RWS_10' in df_precip.columns:
                        existing_row['RWS_10'] = df_precip.loc[idx, 'RWS_10']
                    if 'RWS_IND_10' in df_precip.columns:
                        existing_row['RWS_IND_10'] = df_precip.loc[idx, 'RWS_IND_10']
                
                print(f"  Processed: {os.path.basename(precip_file)}")
                
            except Exception as e:
                print(f"  Error processing {precip_file}: {e}")
        
        # 4. Process 10-minute solar data from station 04928
        print("\n4. Processing 10-minute solar data from station 04928...")
        solar_files = glob.glob("results/weather_data/rawData/10min_solar/produkt_zehn_min_sd_*.txt")
        
        for solar_file in solar_files:
            try:
                # Read solar file
                df_solar = pd.read_csv(solar_file, sep=';', encoding='latin1', header=0)
                
                print(f"  Solar columns: {list(df_solar.columns)}")
                
                # Extract timestamp
                if 'MESS_DATUM' in df_solar.columns:
                    df_solar['timestamp'] = pd.to_datetime(df_solar['MESS_DATUM'], format='%Y%m%d%H%M')
                
                # Extract SD_10 and GS_10 from 10-minute solar data
                for idx, row in df_solar.iterrows():
                    timestamp = df_solar.loc[idx, 'timestamp']
                    
                    # Find existing row with same timestamp or create new one
                    existing_row = None
                    for data_row in combined_data:
                        if data_row['timestamp'] == timestamp:
                            existing_row = data_row
                            break
                    
                    if existing_row is None:
                        existing_row = {'timestamp': timestamp}
                        combined_data.append(existing_row)
                    
                    # Extract solar parameters from 10-minute solar data
                    if 'SD_10' in df_solar.columns:
                        existing_row['SD_10'] = df_solar.loc[idx, 'SD_10']  # 10-minute sunshine duration
                    if 'GS_10' in df_solar.columns:
                        # Convert from J/cm² to W/m²: 1 J/cm² = 10000 W/m²
                        gs_value = df_solar.loc[idx, 'GS_10']
                        if pd.notna(gs_value) and gs_value != -999:
                            existing_row['GS_10'] = gs_value * 10000 / 600  # Convert to W/m² (10-minute interval)
                        else:
                            existing_row['GS_10'] = None
                
                print(f"  Processed: {os.path.basename(solar_file)}")
                
            except Exception as e:
                print(f"  Error processing {solar_file}: {e}")
        
        if not solar_files:
            print(f"  Warning: No 10-minute solar data files found in results/weather_data/rawData/10min_solar/")
        
        # Create final DataFrame
        if combined_data:
            df_combined = pd.DataFrame(combined_data)
            
            # Sort by timestamp
            df_combined = df_combined.sort_values('timestamp').reset_index(drop=True)
            
            # Convert UTC timestamps to German local time (CET/CEST)
            print(f"\nConverting UTC timestamps to German local time...")
            df_combined = self.convert_utc_timestamps_in_dataframe(df_combined)
            
            # Save to CSV
            output_path = "results/weather_data/rawData/combined_weatherData_raw.csv"
            df_combined.to_csv(output_path, index=False)
            
            print(f"\nSUCCESS: Combined CSV created: {output_path}")
            print(f"   Records: {len(df_combined)}")
            print(f"   Columns: {list(df_combined.columns)} (time features will be added during interpolation)")
            print(f"   Time range: {df_combined['timestamp'].min()} to {df_combined['timestamp'].max()}")
            print(f"   Data sources: 10-minute weather from station 04931, 10-minute solar from station 04928")
            print(f"   Timezone: German local time (CET/CEST) with automatic DST handling")
            
            return df_combined
        else:
            print("ERROR: No data found to combine")
            return pd.DataFrame()

    def validate_raw_weather_data(self):
        """Validate and clean raw weather data once, return validated DataFrame"""
        print("="*60)
        print("VALIDATING RAW WEATHER DATA")
        print("="*60)
        
        # Read raw CSV file
        raw_file = "results/weather_data/rawData/combined_weatherData_raw.csv"
        
        if not os.path.exists(raw_file):
            print(f"ERROR: Raw CSV file not found: {raw_file}")
            print("Please run process_txt_files_to_csv() first")
            return pd.DataFrame()
        
        try:
            # Read raw data
            df_raw = pd.read_csv(raw_file)
            print(f"Loaded raw data: {len(df_raw)} records")
            print(f"Columns: {list(df_raw.columns)}")
            
            # Convert timestamp to datetime
            df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])
            
            # Get time range
            start_time = df_raw['timestamp'].min()
            end_time = df_raw['timestamp'].max()
            print(f"Time range: {start_time} to {end_time}")
            
            # Validate and clean raw data
            print(f"\nValidating raw data...")
            df_validated = self.validate_and_interpolate_weather_data(df_raw, "historical")
            
            # Remove duplicate timestamps
            df_validated = df_validated.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
            print(f"Removed duplicate timestamps, remaining records: {len(df_validated)}")
            
            print(f"\nSUCCESS: Raw weather data validated and cleaned")
            print(f"   Records: {len(df_validated)}")
            print(f"   Time range: {df_validated['timestamp'].min()} to {df_validated['timestamp'].max()}")
            
            return df_validated
            
        except Exception as e:
            print(f"ERROR: Error during validation: {e}")
            return pd.DataFrame()

    def create_10min_resolution(self, df_validated):
        """Create 10min resolution from validated data"""
        print("="*60)
        print("CREATING 10MIN RESOLUTION")
        print("="*60)
        
        if df_validated.empty:
            print("ERROR: No validated data provided")
            return pd.DataFrame()
        
        try:
            # Copy validated data for 10min resolution
            df_10min = df_validated.copy()
            
            # Add time features if not present
            need_time = any(col not in df_10min.columns for col in ['hour','minute','weekday','month','day_of_year'])
            if need_time:
                df_10min = self.add_time_features(df_10min, ['hour','minute','weekday','month','day_of_year'])
            
            # Save 10min resolution
            path_10min = "results/weather_data/cleanData/combined_weatherData-10min.csv"
            df_10min_out = df_10min.copy()
            df_10min_out['timestamp'] = df_10min_out['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            df_10min_out.to_csv(path_10min, index=False)
            
            print(f"SUCCESS: 10min resolution created")
            print(f"   Records: {len(df_10min_out)}")
            print(f"   Saved to: {path_10min}")
            
            return df_10min
            
        except Exception as e:
            print(f"ERROR: Error creating 10min resolution: {e}")
            return pd.DataFrame()

    def create_5min_resolution(self, df_validated):
        """Create 5min resolution from validated data"""
        print("="*60)
        print("CREATING 5MIN RESOLUTION")
        print("="*60)
        
        if df_validated.empty:
            print("ERROR: No validated data provided")
            return pd.DataFrame()
        
        try:
            # Get time range from validated data
            start_time = df_validated['timestamp'].min()
            end_time = df_validated['timestamp'].max()
            
            # Create 5-minute timestamp range
            timestamps_5min = pd.date_range(
                start=start_time,
                end=end_time,
                freq='5min'
            )
            
            print(f"Created {len(timestamps_5min)} 5-minute timestamps")
            
            # Create target DataFrame
            df_5min = pd.DataFrame({'timestamp': timestamps_5min})
            
            # Set validated data timestamp as index for interpolation
            df_validated_indexed = df_validated.set_index('timestamp')
            
            # Interpolate each column
            for column in df_validated_indexed.columns:
                if column in ['TT_10', 'RF_10', 'RWS_10', 'GS_10']:
                    # Linear interpolation for continuous variables (10-min to 5-min)
                    df_5min[column] = np.interp(
                        df_5min['timestamp'].astype(np.int64),
                        df_validated_indexed.index.astype(np.int64),
                        df_validated_indexed[column].values,
                        left=df_validated_indexed[column].iloc[0],
                        right=df_validated_indexed[column].iloc[-1]
                    )
                elif column in ['V_N', 'RWS_IND_10', 'SD_10']:
                    # Forward fill for discrete/categorical variables
                    df_5min[column] = df_validated_indexed[column].reindex(
                        df_5min['timestamp'], method='ffill'
                    ).values
                else:
                    # Forward fill for other variables
                    df_5min[column] = df_validated_indexed[column].reindex(
                        df_5min['timestamp'], method='ffill'
                    ).values
            
            # Add time features
            df_5min = self.add_time_features(df_5min, ['hour', 'minute', 'weekday', 'month', 'day_of_year'])
            
            # Save 5min resolution
            path_5min = "results/weather_data/cleanData/combined_weatherData-5min.csv"
            df_5min_out = df_5min.copy()
            df_5min_out['timestamp'] = df_5min_out['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            df_5min_out.to_csv(path_5min, index=False)
            
            print(f"SUCCESS: 5min resolution created")
            print(f"   Records: {len(df_5min_out)} (interpolated from {len(df_validated)} 10-minute records)")
            print(f"   Saved to: {path_5min}")
            
            return df_5min
            
        except Exception as e:
            print(f"ERROR: Error creating 5min resolution: {e}")
            return pd.DataFrame()

    def create_1h_resolution(self, df_validated):
        """Create 1h resolution from validated data"""
        print("="*60)
        print("CREATING 1H RESOLUTION")
        print("="*60)
        
        if df_validated.empty:
            print("ERROR: No validated data provided")
            return pd.DataFrame()
        
        try:
            # Copy validated data for 1h resolution
            df_1h = df_validated.copy()
            
            # Group by hour bins - each XX:00 timestamp represents the previous hour (XX-1:10 to XX:00)
            df_1h['ts1h'] = df_1h['timestamp'].dt.ceil('H')
            
            # Define aggregation rules
            weather_cols = [c for c in df_1h.columns if c in ['TT_10','RF_10','V_N','RWS_10','RWS_IND_10','GS_10','SD_10']]
            time_cols = [c for c in ['hour','weekday','month','day_of_year'] if c in df_1h.columns]
            agg = {col: 'mean' for col in weather_cols}
            agg.update({col: 'first' for col in time_cols})
            
            # Group by hour and aggregate
            df_1h = df_1h.groupby('ts1h').agg(agg).reset_index().rename(columns={'ts1h':'timestamp'})
            df_1h['minute'] = 0
            
            # Add time features if not present
            need_time = any(col not in df_1h.columns for col in ['hour','weekday','month','day_of_year'])
            if need_time:
                df_1h = self.add_time_features(df_1h, ['hour','weekday','month','day_of_year'])
            
            # Save 1h resolution
            path_1h = "results/weather_data/cleanData/combined_weatherData-1h.csv"
            df_1h_out = df_1h.copy()
            df_1h_out['timestamp'] = df_1h_out['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            df_1h_out.to_csv(path_1h, index=False)
            
            print(f"SUCCESS: 1h resolution created")
            print(f"   Records: {len(df_1h_out)} (aggregated from {len(df_validated)} 10-minute records)")
            print(f"   Saved to: {path_1h}")
            print(f"   Note: 1h timestamps represent the PAST hour (e.g., 02:00 = 01:10-02:00)")
            
            return df_1h
            
        except Exception as e:
            print(f"ERROR: Error creating 1h resolution: {e}")
            return pd.DataFrame()

    def process_weather_resolutions(self):
        """Main function: validate once, then create all three resolutions"""
        print("="*80)
        print("PROCESSING WEATHER DATA RESOLUTIONS")
        print("="*80)
        
        try:
            # Step 1: Validate raw data once
            df_validated = self.validate_raw_weather_data()
            if df_validated.empty:
                print("ERROR: Failed to validate raw data")
                return False
            
            # Step 2: Create all three resolutions from validated data
            print("\n" + "="*80)
            print("CREATING ALL THREE RESOLUTIONS")
            print("="*80)
            
            # Create 10min resolution
            df_10min = self.create_10min_resolution(df_validated)
            if df_10min.empty:
                print("ERROR: Failed to create 10min resolution")
                return False
            
            # Create 5min resolution
            df_5min = self.create_5min_resolution(df_validated)
            if df_5min.empty:
                print("ERROR: Failed to create 5min resolution")
                return False
            
            # Create 1h resolution
            df_1h = self.create_1h_resolution(df_validated)
            if df_1h.empty:
                print("ERROR: Failed to create 1h resolution")
                return False
            
            # Step 3: Create legacy file (5min resolution)
            legacy_path = "results/weather_data/cleanData/combined_weatherData_interpol.csv"
            df_5min_out = df_5min.copy()
            df_5min_out['timestamp'] = df_5min_out['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            df_5min_out.to_csv(legacy_path, index=False)
            print(f"Legacy file saved: {legacy_path} (5min resolution)")
            
            # Step 4: Create visualization
            self.create_parameter_visualization(df_5min, "historical")
            
            print("\n" + "="*80)
            print("WEATHER DATA RESOLUTIONS COMPLETED SUCCESSFULLY!")
            print("="*80)
            print("Generated files:")
            print(f"   10min: results/weather_data/cleanData/combined_weatherData-10min.csv ({len(df_10min)} records)")
            print(f"   5min:  results/weather_data/cleanData/combined_weatherData-5min.csv ({len(df_5min)} records)")
            print(f"   1h:    results/weather_data/cleanData/combined_weatherData-1h.csv ({len(df_1h)} records)")
            print(f"   Legacy: {legacy_path} ({len(df_5min)} records)")
            
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to process weather resolutions: {e}")
            return False

    def download_and_process_forecast(self):
        """Download MOSMIX-L forecast data and create forecast CSV"""
        print("="*60)
        print("DOWNLOADING AND PROCESSING FORECAST DATA")
        print("="*60)
        
        # Create forecast directory
        forecast_dir = "results/weather_data/forecast_data/rawData"
        os.makedirs(forecast_dir, exist_ok=True)
        print(f"Created forecast directory: {forecast_dir}")
        
        # MOSMIX-L URL for Stuttgart (Station 10738)
        station_id = "10738"  # Stuttgart for MOSMIX-L
        
        # Get available MOSMIX-L files and select the oldest one
        base_url = f"https://opendata.dwd.de/weather/local_forecasts/mos/MOSMIX_L/single_stations/{station_id}/kml/"
        
        try:
            print(f"Checking available MOSMIX-L files for station {station_id}...")
            
            # Get the directory listing
            response = requests.get(base_url)
            response.raise_for_status()
            
            # Parse the HTML to find KMZ files
            import re
            kmz_files = re.findall(r'href="(MOSMIX_L_.*?\.kmz)"', response.text)
            
            if not kmz_files:
                print("ERROR: No KMZ files found in directory listing")
                return pd.DataFrame()
            
            # Sort files to find the oldest one (non-LATEST)
            # Filter out LATEST files and sort by timestamp
            non_latest_files = [f for f in kmz_files if 'LATEST' not in f]
            
            if non_latest_files:
                # Sort by timestamp (assuming format like MOSMIX_L_YYYYMMDD_HHMM_10738.kmz)
                # Extract timestamp from filename for better sorting
                def extract_timestamp(filename):
                    # Try to extract YYYYMMDD_HHMM from filename
                    import re
                    match = re.search(r'MOSMIX_L_(\d{8}_\d{4})_', filename)
                    if match:
                        return match.group(1)
                    return filename  # Fallback to filename if no timestamp found
                
                # Sort by extracted timestamp
                non_latest_files.sort(key=extract_timestamp)
                # Take the 4th oldest file (index 3, since indexing starts at 0)
                if len(non_latest_files) >= 4:
                    fourth_oldest_file = non_latest_files[3]  # 4th oldest (index 3)
                    print(f"Found 4th oldest available file: {fourth_oldest_file}")
                    print(f"Available files (sorted by timestamp): {non_latest_files[:10]}...")  # Show first 10 files
                    print(f"Oldest file: {non_latest_files[0]}")
                    print(f"2nd oldest file: {non_latest_files[1]}")
                    print(f"3rd oldest file: {non_latest_files[2]}")
                    print(f"4th oldest file (selected): {fourth_oldest_file}")
                elif len(non_latest_files) >= 1:
                    # Fallback to oldest if less than 4 files available
                    fourth_oldest_file = non_latest_files[0]
                    print(f"WARNING: Less than 4 files available, using oldest: {fourth_oldest_file}")
                    print(f"Available files: {non_latest_files}")
                else:
                    print("ERROR: No timestamped files found")
                    return pd.DataFrame()
            else:
                # Fallback to LATEST if no timestamped files found
                latest_files = [f for f in kmz_files if 'LATEST' in f]
                if latest_files:
                    oldest_file = latest_files[0]
                    print(f"WARNING: No timestamped files found, using LATEST: {oldest_file}")
                else:
                    print("ERROR: No suitable KMZ files found")
                    return pd.DataFrame()
            
            # Construct full URL
            url = base_url + fourth_oldest_file
            
            print(f"Downloading MOSMIX-L forecast: {fourth_oldest_file}")
            response = requests.get(url)
            response.raise_for_status()
            
            # Extract KMZ file and save contents
            kmz_filename = fourth_oldest_file
            kmz_path = f"{forecast_dir}/{kmz_filename}"
            
            # Save KMZ temporarily
            with open(kmz_path, 'wb') as f:
                f.write(response.content)
            
            print(f"SUCCESS: Downloaded: {kmz_path}")
            
            # Extract KMZ contents
            with zipfile.ZipFile(kmz_path, 'r') as kmz_file:
                # List contents
                print(f"KMZ contents:")
                for file_info in kmz_file.filelist:
                    print(f"  - {file_info.filename} ({file_info.file_size} bytes)")
                
                # Extract all files
                kmz_file.extractall(forecast_dir)
                print(f"SUCCESS: Extracted KMZ contents to: {forecast_dir}")
            
            # Delete temporary KMZ file
            os.remove(kmz_path)
            print(f"Removed temporary KMZ file")
            
            # Process extracted files
            df_forecast = self.extract_forecast_from_kmz(forecast_dir)
            
            if not df_forecast.empty:
                # Save forecast CSV
                csv_path = f"{forecast_dir}/combined_forecast_weatherData_raw.csv"
                df_forecast.to_csv(csv_path, index=False)
                print(f"\nSUCCESS: Forecast CSV created: {csv_path}")
                print(f"   Records: {len(df_forecast)}")
                print(f"   Columns: {list(df_forecast.columns)} (time features will be added during interpolation)")
                print(f"   Time range: {df_forecast['timestamp'].min()} to {df_forecast['timestamp'].max()}")
                return df_forecast
            else:
                print("ERROR: No forecast data extracted")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"ERROR: Error downloading forecast: {e}")
            return pd.DataFrame()
    
    def extract_forecast_from_kmz(self, forecast_dir):
        """Extract forecast data from extracted KMZ directory using GRIB2 files"""
        print(f"Extracting forecast data from extracted KMZ files...")
        
        try:
            # List extracted files
            extracted_files = os.listdir(forecast_dir)
            print(f"Extracted files: {extracted_files}")
            
            # Find GRIB2 file
            grb2_file = None
            for file in extracted_files:
                if file.endswith('.grb2') or file.endswith('.grib2'):
                    grb2_file = os.path.join(forecast_dir, file)
                    break
            
            if not grb2_file:
                print("  No GRIB2 file found in extracted KMZ")
                print("  Creating fallback forecast data from KML metadata...")
                
                # Try to extract forecast data from KML file as fallback
                kml_file = None
                for file in extracted_files:
                    if file.endswith('.kml'):
                        kml_file = os.path.join(forecast_dir, file)
                        break
                
                if kml_file:
                    return self.extract_forecast_from_kml(kml_file)
                else:
                    print("  No KML file found either")
                    return pd.DataFrame()
            
            print(f"  Processing GRIB2 file: {os.path.basename(grb2_file)}")
            
            # Open GRIB2 file
            grbs = pygrib.open(grb2_file)
            
            # Get available parameters
            available_params = []
            for grb in grbs:
                param_name = grb.shortName
                level_type = grb.levelType
                level = grb.level
                forecast_time = grb.forecastTime
                
                param_info = {
                    'shortName': param_name,
                    'levelType': level_type,
                    'level': level,
                    'forecastTime': forecast_time,
                    'units': grb.units,
                    'name': grb.name
                }
                available_params.append(param_info)
            
            print(f"  Available parameters in GRIB2:")
            for param in available_params[:10]:  # Show first 10
                print(f"    {param['shortName']} ({param['name']}) - Level: {param['levelType']} {param['level']} - Time: +{param['forecastTime']}h - Units: {param['units']}")
            
            # Extract forecast data for specific parameters
            forecast_data = []
            
            # Get unique forecast times
            forecast_times = sorted(list(set([p['forecastTime'] for p in available_params])))
            print(f"  Forecast times: {forecast_times}")
            
            # Calculate base timestamp (analysis time)
            base_time = datetime.now().replace(minute=0, second=0, microsecond=0)
            
            for fc_time in forecast_times:
                # Calculate timestamp for this forecast
                timestamp = base_time + timedelta(hours=fc_time)
                
                # Extract parameters for this forecast time
                temp_2m = None
                humidity_2m = None
                cloud_cover = None
                precipitation = None
                
                # Reset file pointer
                grbs.seek(0)
                
                for grb in grbs:
                    if grb.forecastTime == fc_time:
                        param_name = grb.shortName
                        level_type = grb.levelType
                        level = grb.level
                        
                        # Get data for this parameter
                        data = grb.values
                        
                        # For now, use the first grid point (nearest to station)
                        # In a real implementation, you would interpolate to station coordinates
                        value = data[0, 0] if data.size > 0 else None
                        
                        # Map GRIB2 parameters to our required parameters
                        if param_name == '2t' and level_type == 'sfc':  # 2m temperature
                            temp_2m = value - 273.15 if value is not None else None  # Convert K to °C
                        elif param_name == 'r' and level_type == 'sfc':  # Relative humidity
                            humidity_2m = value if value is not None else None
                        elif param_name == 'tcc' and level_type == 'sfc':  # Total cloud cover
                            cloud_cover = value if value is not None else None
                        elif param_name == 'tp' and level_type == 'sfc':  # Total precipitation
                            precipitation = value * 1000 if value is not None else None  # Convert m to mm
                
                # Create forecast record
                forecast_record = {
                    'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    'TT_TU': round(temp_2m, 1) if temp_2m is not None else None,
                    'RF_TU': round(humidity_2m, 1) if humidity_2m is not None else None,
                    'V_N': int(cloud_cover) if cloud_cover is not None else None,
                    'R1': round(precipitation, 1) if precipitation is not None else None,
                    'RS_IND': 1 if precipitation and precipitation > 0 else 0,
                    'WRTR': "Regen" if precipitation and precipitation > 0 else "Kein Niederschlag",
                    'is_forecast': True
                }
                
                forecast_data.append(forecast_record)
            
            grbs.close()
            
            df_forecast = pd.DataFrame(forecast_data)
            
            # Remove rows with all None values
            df_forecast = df_forecast.dropna(subset=['TT_TU', 'RF_TU', 'V_N', 'R1'], how='all')
            
            print(f"  Extracted {len(df_forecast)} forecast records from GRIB2")
            if not df_forecast.empty:
                print(f"  Parameters: TT_TU, RF_TU, V_N, R1, RS_IND, WRTR")
                print(f"  Temperature range: {df_forecast['TT_TU'].min():.1f}°C to {df_forecast['TT_TU'].max():.1f}°C")
                print(f"  Forecast period: {df_forecast['timestamp'].min()} to {df_forecast['timestamp'].max()}")
                print(f"  Columns: {list(df_forecast.columns)} (time features will be added during interpolation)")
            
            return df_forecast
            
        except Exception as e:
            print(f"  Error extracting forecast data: {e}")
            print(f"  This might be due to missing pygrib library or incompatible GRIB2 format")
            return pd.DataFrame()
    
    def extract_forecast_from_kml(self, kml_file):
        """Extract forecast data from KML file as fallback"""
        print(f"  Extracting forecast data from KML: {os.path.basename(kml_file)}")
        
        try:
            # Read KML file
            with open(kml_file, 'r', encoding='utf-8') as f:
                kml_content = f.read()
            
            # Parse KML content to extract forecast data
            # This is a simplified parser for DWD MOSMIX-L KML format
            import re
            
            # Extract timestamps from ForecastTimeSteps
            timestamp_pattern = r'<dwd:TimeStep>(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\.000Z</dwd:TimeStep>'
            timestamps = re.findall(timestamp_pattern, kml_content)
            
            # If no timestamps found, try without .000Z
            if not timestamps:
                timestamp_pattern = r'<dwd:TimeStep>(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})</dwd:TimeStep>'
                timestamps = re.findall(timestamp_pattern, kml_content)
            
            # Extract forecast data from dwd:Forecast elements with proper mapping
            # TT_10: TTT (Temperature 2m above surface) - Convert K to °C
            ttt_pattern = r'<dwd:Forecast dwd:elementName="TTT">\s*<dwd:value>([^<]+)</dwd:value>'
            ttt_match = re.search(ttt_pattern, kml_content)
            temperatures = []
            if ttt_match:
                temp_values = ttt_match.group(1).strip().split()
                temperatures = [float(val) - 273.15 for val in temp_values if val != '-']  # Convert K to °C
            else:
                temperatures = []
            
            # RF_10: Calculate from Td (Dewpoint 2m above surface) and TTT
            td_pattern = r'<dwd:Forecast dwd:elementName="Td">\s*<dwd:value>([^<]+)</dwd:value>'
            td_match = re.search(td_pattern, kml_content)
            humidities = []
            if td_match and temperatures:
                td_values = td_match.group(1).strip().split()
                td_temps = [float(val) - 273.15 for val in td_values if val != '-']  # Convert K to °C
                
                # Calculate relative humidity using Magnus formula
                for i, (temp, td) in enumerate(zip(temperatures, td_temps)):
                    if temp != '-' and td != '-':
                        # Magnus formula: RH = 100 * exp((17.625 * Td) / (243.04 + Td)) / exp((17.625 * T) / (243.04 + T))
                        es_td = 6.1094 * np.exp((17.625 * td) / (243.04 + td))
                        es_t = 6.1094 * np.exp((17.625 * temp) / (243.04 + temp))
                        rh = 100 * es_td / es_t
                        humidities.append(min(100, max(0, rh)))
                    else:
                        humidities.append(None)
            else:
                humidities = []
            
            # V_N: N (Cloud cover) - Convert from 0-100% to 0-8 scale
            n_pattern = r'<dwd:Forecast dwd:elementName="N">\s*<dwd:value>([^<]+)</dwd:value>'
            n_match = re.search(n_pattern, kml_content)
            cloudiness = []
            if n_match:
                n_values = n_match.group(1).strip().split()
                # Convert from 0-100% to 0-8 scale (rounded)
                cloudiness = [round(float(val) * 8 / 100) if val != '-' else None for val in n_values]
            else:
                cloudiness = []
            
            # RWS_10: RR1c (Precipitation amount) - Already in mm
            rr1_pattern = r'<dwd:Forecast dwd:elementName="RR1c">\s*<dwd:value>([^<]+)</dwd:value>'
            rr1_match = re.search(rr1_pattern, kml_content)
            precipitation = []
            if rr1_match:
                rr1_values = rr1_match.group(1).strip().split()
                # DWD precipitation is already in mm
                precipitation = [float(val) if val != '-' else 0.0 for val in rr1_values]
            else:
                precipitation = []
            
            # RWS_IND_10: Derived from RR1 (precipitation indicator)
            precip_indicator = []
            if precipitation:
                precip_indicator = [1 if val > 0 else 0 for val in precipitation]
            else:
                precip_indicator = []
            
            # GS_10: Rad1h (Global irradiance within the last hour) - Convert from kJ/m² to W/m²
            rad1h_pattern = r'<dwd:Forecast dwd:elementName="Rad1h">\s*<dwd:value>([^<]+)</dwd:value>'
            rad1h_match = re.search(rad1h_pattern, kml_content)
            irradiance = []
            if rad1h_match:
                rad1h_values = rad1h_match.group(1).strip().split()
                # Convert from kJ/m² to W/m²: 1 kJ/m² = 1000 J/m², divide by 3600s (1 hour) to get W/m²
                irradiance = [float(val) * 1000 / 3600 if val != '-' else 0.0 for val in rad1h_values]
            else:
                irradiance = []
            
            # SD_10: SunD1 (Sunshine duration during the last hour) - Already in minutes
            sund1_pattern = r'<dwd:Forecast dwd:elementName="SunD1">\s*<dwd:value>([^<]+)</dwd:value>'
            sund1_match = re.search(sund1_pattern, kml_content)
            sunshine_duration = []
            if sund1_match:
                sund1_values = sund1_match.group(1).strip().split()
                # DWD sunshine duration is already in minutes
                sunshine_duration = [float(val) if val != '-' else 0.0 for val in sund1_values]
            else:
                sunshine_duration = []
            

            

            
            print(f"    Found {len(timestamps)} forecast timestamps")
            
            if not timestamps:
                print("    No forecast data found in KML")
                return pd.DataFrame()
            
            # Create forecast DataFrame - only use available data
            forecast_data = []
            max_length = max(len(timestamps), len(temperatures), len(humidities), len(cloudiness), len(precipitation), len(precip_indicator), len(irradiance), len(sunshine_duration))
            
            print(f"    Available data lengths:")
            print(f"      Timestamps: {len(timestamps)}")
            print(f"      Temperatures: {len(temperatures)}")
            print(f"      Humidities: {len(humidities)}")
            print(f"      Cloudiness: {len(cloudiness)}")
            print(f"      Precipitation: {len(precipitation)}")
            print(f"      Precip Indicator: {len(precip_indicator)}")
            print(f"      Irradiance: {len(irradiance)} (from {'Rad1h' if rad1h_match else 'none'})")
            print(f"      Sunshine Duration: {len(sunshine_duration)}")
            print(f"    Using maximum length: {max_length}")
            
            # Create forecast data entries
            for i in range(max_length):
                forecast_data.append({
                    'timestamp': timestamps[i] if i < len(timestamps) else None,
                    'TT_10': round(temperatures[i], 1) if i < len(temperatures) and temperatures[i] is not None else None,
                    'RF_10': round(humidities[i], 1) if i < len(humidities) and humidities[i] is not None else None,
                    'V_N': cloudiness[i] if i < len(cloudiness) and cloudiness[i] is not None else None,
                    'RWS_10': round(precipitation[i], 1) if i < len(precipitation) and precipitation[i] is not None else None,
                    'RWS_IND_10': precip_indicator[i] if i < len(precip_indicator) and precip_indicator[i] is not None else None,
                    'GS_10': round(irradiance[i], 1) if i < len(irradiance) and irradiance[i] is not None else None,
                    'SD_10': round(sunshine_duration[i], 1) if i < len(sunshine_duration) and sunshine_duration[i] is not None else None
                })
            
            df_forecast = pd.DataFrame(forecast_data)
            
            # Remove rows with all None values
            df_forecast = df_forecast.dropna(subset=['TT_10', 'RF_10', 'V_N', 'RWS_10'], how='all')
            
            # Convert UTC timestamps to German local time (CET/CEST)
            print(f"    Converting forecast UTC timestamps to German local time...")
            
            # Ensure timestamps are datetime objects before conversion
            df_forecast['timestamp'] = pd.to_datetime(df_forecast['timestamp'])
            df_forecast = self.convert_utc_timestamps_in_dataframe(df_forecast)
            
            print(f"    Extracted {len(df_forecast)} forecast records from KML")
            if not df_forecast.empty:
                print(f"    Temperature range: {df_forecast['TT_10'].min():.1f}°C to {df_forecast['TT_10'].max():.1f}°C")
                print(f"    Forecast period: {df_forecast['timestamp'].min()} to {df_forecast['timestamp'].max()}")
                print(f"    Columns: {list(df_forecast.columns)} (time features will be added during interpolation)")
                print(f"    Timezone: German local time (CET/CEST) with automatic DST handling")
            
            return df_forecast
            
        except Exception as e:
            print(f"    Error extracting from KML: {e}")
            return pd.DataFrame()

    def interpolate_forecast_to_5min(self):
        """Interpolate forecast CSV data to 5-minute resolution"""
        print("="*60)
        print("INTERPOLATING FORECAST TO 5-MINUTE RESOLUTION")
        print("="*60)
        
        # Read forecast CSV file
        forecast_file = "results/weather_data/forecast_data/rawData/combined_forecast_weatherData_raw.csv"
        
        if not os.path.exists(forecast_file):
            print(f"ERROR: Forecast CSV file not found: {forecast_file}")
            print("Please run download_and_process_forecast() first")
            return pd.DataFrame()
        
        try:
            # Read forecast data
            df_forecast = pd.read_csv(forecast_file)
            print(f"Loaded forecast data: {len(df_forecast)} records")
            print(f"Columns: {list(df_forecast.columns)}")
            
            # Convert timestamp to datetime
            df_forecast['timestamp'] = pd.to_datetime(df_forecast['timestamp'])
            
            # Get time range
            start_time = df_forecast['timestamp'].min()
            end_time = df_forecast['timestamp'].max()
            print(f"Forecast time range: {start_time} to {end_time}")
            
            # Create 5-minute timestamp range
            timestamps_5min = pd.date_range(
                start=start_time,
                end=end_time,
                freq='5min'
            )
            
            print(f"Created {len(timestamps_5min)} 5-minute timestamps")
            
            # Create target DataFrame
            df_5min = pd.DataFrame({'timestamp': timestamps_5min})
            
            # Validate and clean forecast data BEFORE interpolation
            print(f"\nValidating forecast data before interpolation...")
            df_forecast = self.validate_and_interpolate_weather_data(df_forecast, "forecast")
            
            # Set timestamp as index for interpolation
            df_forecast = df_forecast.set_index('timestamp')
            
            # Interpolate each column
            for column in df_forecast.columns:
                if column in ['TT_10', 'RF_10', 'RWS_10', 'GS_10']:
                    # Linear interpolation for continuous variables (hourly to 5-min)
                    df_5min[column] = np.interp(
                        df_5min['timestamp'].astype(np.int64),
                        df_forecast.index.astype(np.int64),
                        df_forecast[column].values,
                        left=df_forecast[column].iloc[0],
                        right=df_forecast[column].iloc[-1]
                    )
                elif column in ['V_N', 'RWS_IND_10', 'SD_10']:
                    # Forward fill for discrete/categorical variables
                    df_5min[column] = df_forecast[column].reindex(
                        df_5min['timestamp'], method='ffill'
                    ).values
                else:
                    # Forward fill for other variables
                    df_5min[column] = df_forecast[column].reindex(
                        df_5min['timestamp'], method='ffill'
                    ).values
            
            # Add time features to the interpolated DataFrame
            df_5min = self.add_time_features(df_5min, ['hour', 'minute', 'weekday', 'month', 'day_of_year'])

            # Format timestamp back to string
            df_5min['timestamp'] = df_5min['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Create and save all three resolution variants for forecast data
            self._save_forecast_resolution_variants(df_forecast, df_5min)
            
            print(f"\nSUCCESS: Forecast data resolution variants created")
            print(f"   5min records: {len(df_5min)} (interpolated from {len(df_forecast)} hourly records)")
            print(f"   Columns: {list(df_5min.columns)} (including time features: hour, minute, weekday, month, day_of_year)")
            print(f"   Time range: {df_5min['timestamp'].min()} to {df_5min['timestamp'].max()}")
            
            # Show final data quality
            print(f"\nForecast data quality:")
            for column in df_5min.columns:
                if column != 'timestamp':
                    missing = df_5min[column].isna().sum()
                    total = len(df_5min)
                    percentage = (missing / total) * 100
                    print(f"   {column}: {missing} missing ({percentage:.1f}%)")
            
            # Create visualization
            self.create_parameter_visualization(df_5min, "forecast")
            
            return df_5min
            
        except Exception as e:
            print(f"ERROR: Error during forecast interpolation: {e}")
            return pd.DataFrame()

    def create_parameter_visualization(self, df, data_type="historical"):
        """Create interactive visualization of all weather parameters"""
        print("="*60)
        print(f"CREATING {data_type.upper()} PARAMETER VISUALIZATION")
        print("="*60)
        
        if not PLOTLY_AVAILABLE:
            print("ERROR: Plotly not available. Please install: pip install plotly kaleido")
            return None
            
        try:
            
            # Convert timestamp to datetime if it's a string
            if df['timestamp'].dtype == 'object':
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Define parameter groups for visualization
            parameter_groups = {
                'Temperature & Humidity': ['TT_10', 'RF_10'],
                'Precipitation': ['RWS_10', 'RWS_IND_10'],
                'Cloudiness': ['V_N'],
                'Solar': ['GS_10', 'SD_10']
            }

            # Flatten the list of all parameters to plot
            all_parameters_to_plot = []
            for group_name, params in parameter_groups.items():
                all_parameters_to_plot.extend(params)
            
            # Filter out parameters that are not in the DataFrame
            all_parameters_to_plot = [p for p in all_parameters_to_plot if p in df.columns]

            if len(all_parameters_to_plot) == 0:
                print("ERROR: No parameters found for visualization from the combined data.")
                return
            
            print(f"Creating visualization for {len(all_parameters_to_plot)} parameters: {all_parameters_to_plot}")
            
            # Create subplots
            fig = make_subplots(
                rows=len(all_parameters_to_plot), 
                cols=1,
                subplot_titles=all_parameters_to_plot,
                vertical_spacing=0.05,
                shared_xaxes=True
            )
            
            # Color palette for parameters
            colors = px.colors.qualitative.Set1
            
            # Add traces for each parameter
            for i, column in enumerate(all_parameters_to_plot):
                color = colors[i % len(colors)]
                
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df[column],
                        mode='lines',
                        name=column,
                        line=dict(color=color, width=1.5),
                        showlegend=True,
                        visible=True if i < 3 else 'legendonly'  # Show first 3 by default
                    ),
                    row=i+1, 
                    col=1
                )
                
                # Update y-axis labels
                fig.update_yaxes(title_text=column, row=i+1, col=1)
            
            # Update layout
            fig.update_layout(
                title=f"{data_type.title()} Weather Parameters - 5-Minute Resolution",
                height=300 * len(all_parameters_to_plot),  # Dynamic height based on number of parameters
                width=1200,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                hovermode='x unified'
            )
            
            # Update x-axis for all subplots
            fig.update_xaxes(title_text="Time", row=len(all_parameters_to_plot), col=1)
            
            # Create output directory
            output_dir = f"results/weather_data/visualizations"
            os.makedirs(output_dir, exist_ok=True)
            
            # Save as PNG
            png_path = f"{output_dir}/{data_type}_parameters_visualization.png"
            fig.write_image(png_path, width=1200, height=300 * len(all_parameters_to_plot))
            print(f"SUCCESS: PNG saved: {png_path}")
            
            # Save as interactive HTML
            html_path = f"{output_dir}/{data_type}_parameters_visualization.html"
            fig.write_html(html_path)
            print(f"SUCCESS: Interactive HTML saved: {html_path}")
            
            # Show data summary
            print(f"\nVisualization summary:")
            print(f"   Parameters: {len(all_parameters_to_plot)}")
            print(f"   Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"   Data points: {len(df)}")
            print(f"   Resolution: 5-minute intervals")
            
            # Parameter statistics
            print(f"\nParameter statistics:")
            for column in all_parameters_to_plot:
                stats = df[column].describe()
                print(f"   {column}:")
                print(f"     Min: {stats['min']:.2f}")
                print(f"     Max: {stats['max']:.2f}")
                print(f"     Mean: {stats['mean']:.2f}")
                print(f"     Std: {stats['std']:.2f}")
            
            return fig
            
        except ImportError as e:
            print(f"ERROR: Required plotting libraries not available: {e}")
            print("Please install: pip install plotly kaleido")
            return None
        except Exception as e:
            print(f"ERROR: Error creating visualization: {e}")
            return None


def main():
    """Main function to run the complete DWD weather data integration pipeline"""
    print("="*80)
    print("DWD WEATHER DATA INTEGRATION PIPELINE WITH SOLAR DATA")
    print("="*80)
    
    # Initialize integrator
    integrator = DWDOptimizedIntegrator(station_id="04931")  # Stuttgart-Echterdingen
    
    try:
        # 1. Download historical weather data
        print("\n" + "="*80)
        print("STEP 1: DOWNLOADING HISTORICAL WEATHER DATA")
        print("="*80)
        
        # Download 10-minute temperature and humidity data
        integrator.get_10min_temperature_humidity("2024-08-01", "2025-07-26")
        
        # Download hourly cloudiness data (still hourly)
        integrator.get_historical_cloudiness("2024-08-01", "2025-07-26")
        
        # Download 10-minute precipitation data
        integrator.get_10min_precipitation("2024-08-01", "2025-07-26")
        
        # Download 10-minute solar data
        integrator.get_10min_solar_data("2024-08-01", "2025-07-26")
        
        # Step 2: Process TXT files to CSV
        print("\n" + "="*60)
        print("STEP 2: PROCESSING TXT FILES TO CSV")
        print("="*60)
        
        integrator.process_txt_files_to_csv()
        
        # Step 3: Process weather data resolutions (5min, 10min, 1h)
        print("\n" + "="*60)
        print("STEP 3: PROCESSING WEATHER DATA RESOLUTIONS")
        print("="*60)
        
        integrator.process_weather_resolutions()
        
        # Step 4: Download and process forecast data
        print("\n" + "="*60)
        print("STEP 4: DOWNLOADING AND PROCESSING FORECAST DATA")
        print("="*60)
        
        integrator.download_and_process_forecast()
        
        # Step 5: Interpolate forecast to 5-minute resolution
        print("\n" + "="*60)
        print("STEP 5: INTERPOLATING FORECAST TO 5-MINUTE RESOLUTION")
        print("="*60)
        
        integrator.interpolate_forecast_to_5min()
        
        # Step 6: Create visualizations
        print("\n" + "="*60)
        print("STEP 6: CREATING VISUALIZATIONS")
        print("="*60)
        
        # Load historical data for visualization
        historical_file = "results/weather_data/cleanData/combined_weatherData_interpol.csv"
        if os.path.exists(historical_file):
            historical_df = pd.read_csv(historical_file)
            integrator.create_parameter_visualization(historical_df, "historical")
        
        # Load forecast data for visualization
        forecast_file = "results/weather_data/forecast_data/rawData/combined_forecast_weatherData_interpol.csv"
        if os.path.exists(forecast_file):
            forecast_df = pd.read_csv(forecast_file)
            integrator.create_parameter_visualization(forecast_df, "forecast")
        
        print("\n" + "="*80)
        print("DWD WEATHER DATA INTEGRATION PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("Generated files:")
        print("  Historical data: results/weather_data/cleanData/combined_weatherData_interpol.csv")
        print("  Forecast data: results/weather_data/forecast_data/rawData/combined_forecast_weatherData_interpol.csv")
        print("  Visualizations: results/weather_data/visualizations/")
        print("\nAvailable parameters:")
        print("  TT_TU: Temperature (°C)")
        print("  RF_TU: Relative Humidity (%)")
        print("  V_N: Cloudiness (0-8 scale)")
        print("  R1: Precipitation height (mm)")
        print("  RS_IND: Precipitation indicator (0/1)")
        print("  Irr: Global irradiance (W/m²) - NEW!")
        print("  SunD1: Sunshine duration (minutes) - NEW!")
        
    except Exception as e:
        print(f"\nERROR: Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main() 
    