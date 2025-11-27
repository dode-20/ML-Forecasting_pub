import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from typing import List, Dict, Tuple, Optional

COLUMN_MAP = {
    "temp": "Temp",
    "ambtemp": "AmbTemp",
    "ambhmd": "AmbHmd",
    "irr": "Irr",
    "datetime": "Datetime",
    "_time": "Datetime",  # InfluxDB Zeitstempel
    "u": "U",  # InfluxDB field name for Voltage
    "i": "I",  # InfluxDB field name for Current
    "p": "P"      # InfluxDB field name for Power (fixed trailing space)
}

def load_all_data(folder="data"):
    dfs = []
    for file in sorted(os.listdir(folder)):
        if file.endswith(".csv"):
            df = pd.read_csv(
                os.path.join(folder, file),
                sep=";",
                decimal=".",
                thousands=",",
                skiprows=0,
                encoding="utf-8"
            )

            # Erste Spalte: Zeit extrahieren und bereinigen
            df = df.rename(columns={df.columns[0]: "time"})
            df["time"] = df["time"].str.extract(r'(\d{2}:\d{2})', expand=False)

            # Datum aus Dateinamen extrahieren
            date_parts = file.split("_")
            date_str = f"{date_parts[2]}_{date_parts[3]}_{date_parts[4].split('.')[0]}"
            df["timestamp"] = pd.to_datetime(date_str + " " + df["time"], format="%Y_%m_%d %H:%M", errors="coerce")

            # PV-Leistung extrahieren
            df["pv_power_w"] = pd.to_numeric(df["PV power generation / Mean values [W]"], errors="coerce")

            dfs.append(df[["timestamp", "pv_power_w"]])

    full_df = pd.concat(dfs).dropna().reset_index(drop=True)

    # Zusatzfeatures berechnen
    full_df["pv_kwh"] = full_df["pv_power_w"] / 1000.0
    full_df["hour"] = full_df["timestamp"].dt.hour + full_df["timestamp"].dt.minute / 60.0
    full_df["day_of_year"] = full_df["timestamp"].dt.dayofyear
    full_df["weekday"] = full_df["timestamp"].dt.weekday

    # Vorjahreswerte berechnen
    full_df["prev_year"] = full_df["timestamp"] - pd.DateOffset(years=1)
    full_df["pv_prev_year"] = full_df.set_index("timestamp")["pv_kwh"].reindex(full_df["prev_year"]).values
    full_df["pv_prev_year"] = pd.Series(full_df["pv_prev_year"]).fillna(0.0)

    return full_df

class DataPreprocessor:
    def __init__(self, features: List[str], output_features: List[str], sequence_length: int = 864, forecast_steps: int = 1, settings: Dict = None):
        self.features = features
        self.output_features = output_features
        self.sequence_length = sequence_length
        self.forecast_steps = forecast_steps
        self.settings = settings or {}
        
        # Override forecast_steps from settings if available
        if self.settings and 'forecast_mode' in self.settings:
            forecast_mode = self.settings['forecast_mode']
            if isinstance(forecast_mode, dict) and 'forecast_steps' in forecast_mode:
                self.forecast_steps = forecast_mode['forecast_steps']
                print(f"[DEBUG] DataPreprocessor: Overriding forecast_steps from settings: {forecast_steps} -> {self.forecast_steps}")
        
        self.feature_scalers = {feature: MinMaxScaler() for feature in features}
        self.output_scalers = {feature: MinMaxScaler() for feature in output_features}
        
    def _map_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        # Map columns using COLUMN_MAP and case-insensitive matching
        col_map = {}
        lower_cols = {col.lower(): col for col in df.columns}
        for key, val in COLUMN_MAP.items():
            if key in lower_cols:
                col_map[lower_cols[key]] = val
        df = df.rename(columns=col_map)
        return df

    def _process_datetime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Konvertiere Datetime-Spalten in numerische Features."""
        processed_data = data.copy()
        
        # Create a copy of the features list so as not to modify it
        current_features = self.features.copy()
        
        for feature in current_features:
            if feature == "Datetime" and feature in processed_data.columns:
                # Konvertiere zu datetime falls es ein String ist
                if processed_data[feature].dtype == 'object':
                    processed_data[feature] = pd.to_datetime(processed_data[feature])
                
                # Extrahiere numerische Features aus Datetime
                processed_data[f"{feature}_hour"] = processed_data[feature].dt.hour + processed_data[feature].dt.minute / 60.0
                processed_data[f"{feature}_day_of_year"] = processed_data[feature].dt.dayofyear
                processed_data[f"{feature}_weekday"] = processed_data[feature].dt.weekday
                
                # Remove the original datetime column
                processed_data = processed_data.drop(columns=[feature])
                
                # Ersetze "Datetime" in der Feature-Liste durch die neuen numerischen Features
                datetime_features = [f"{feature}_hour", f"{feature}_day_of_year", f"{feature}_weekday"]
                self.features = [f for f in self.features if f != feature] + datetime_features
                
                # Aktualisiere die Scaler
                for new_feature in datetime_features:
                    if new_feature not in self.feature_scalers:
                        self.feature_scalers[new_feature] = MinMaxScaler()
        
        return processed_data

    def _fit_scalers(self, data: pd.DataFrame):
        """Fit scalers on the provided data."""
        # Feature scaling
        for feature in self.features:
            if feature in data.columns:
                # Behandle leere/NaN-Werte
                feature_data = data[feature].fillna(0.0)
                if feature_data.dtype == 'object':
                    # Versuche zu konvertieren, falls es Strings sind
                    feature_data = pd.to_numeric(feature_data, errors='coerce').fillna(0.0)
                
                self.feature_scalers[feature].fit(feature_data.to_numpy().reshape(-1, 1))
            else:
                # Create dummy scaler for missing features
                self.feature_scalers[feature] = MinMaxScaler()
                self.feature_scalers[feature].fit([[0.0], [1.0]])
        
        # Output scaling
        for feature in self.output_features:
            if feature in data.columns:
                # Behandle leere/NaN-Werte
                feature_data = data[feature].fillna(0.0)
                if feature_data.dtype == 'object':
                    # Versuche zu konvertieren, falls es Strings sind
                    feature_data = pd.to_numeric(feature_data, errors='coerce').fillna(0.0)
                
                self.output_scalers[feature].fit(feature_data.to_numpy().reshape(-1, 1))
            else:
                # Create dummy scaler for missing features
                self.output_scalers[feature] = MinMaxScaler()
                self.output_scalers[feature].fit([[0.0], [1.0]])

    def fit(self, data: pd.DataFrame):
        """Fit the preprocessor on the provided data."""
        data = self._map_columns(data)
        data.columns = data.columns.str.strip()
        data = self._process_datetime_features(data)
        self._fit_scalers(data)

    def fit_transform(self, data: pd.DataFrame, weather_forecast_data: pd.DataFrame = None, settings: Dict = None) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self._map_columns(data)
        
        # Clean column names (remove trailing spaces) - do this AFTER mapping
        data.columns = data.columns.str.strip()
        
        data = self._process_datetime_features(data)
        
        # DEBUG: Preprocessing analysis
        print(f"[DEBUG] ===== PREPROCESSOR FIT_TRANSFORM ANALYSIS =====")
        print(f"[DEBUG] Input data shape: {data.shape}")
        print(f"[DEBUG] Input data columns: {list(data.columns)}")
        print(f"[DEBUG] Expected features: {self.features}")
        print(f"[DEBUG] Expected outputs: {self.output_features}")
        
        # Check for missing features/outputs
        missing = [f for f in self.features + self.output_features if f not in data.columns]
        if missing:
            print(f"Warning: Missing columns in data: {missing}")
            print(f"[DEBUG] Available columns: {list(data.columns)}")
        
        # Fit scalers on historical data
        self._fit_scalers(data)
        
        # Feature scaling
        scaled_features = {}
        for feature in self.features:
            if feature in data.columns:
                # Behandle leere/NaN-Werte
                feature_data = data[feature].fillna(0.0)
                if feature_data.dtype == 'object':
                    # Versuche zu konvertieren, falls es Strings sind
                    feature_data = pd.to_numeric(feature_data, errors='coerce').fillna(0.0)
                
                # DEBUG: Feature scaling analysis
                if feature in ['hour', 'minute', 'TT_10', 'RF_10', 'GS_10']:  # Sample features
                    print(f"[DEBUG] Feature '{feature}' before scaling:")
                    print(f"[DEBUG]   min: {feature_data.min()}, max: {feature_data.max()}")
                    print(f"[DEBUG]   mean: {feature_data.mean()}, std: {feature_data.std()}")
                
                scaled_features[feature] = self.feature_scalers[feature].transform(
                    feature_data.to_numpy().reshape(-1, 1)
                )
                
                # DEBUG: Feature scaling results
                if feature in ['hour', 'minute', 'TT_10', 'RF_10', 'GS_10']:  # Sample features
                    print(f"[DEBUG] Feature '{feature}' after scaling:")
                    print(f"[DEBUG]   min: {scaled_features[feature].min()}, max: {scaled_features[feature].max()}")
                    print(f"[DEBUG]   mean: {scaled_features[feature].mean()}, std: {scaled_features[feature].std()}")
            else:
                scaled_features[feature] = np.zeros((len(data), 1))
                print(f"[WARN] Feature '{feature}' not found, using zeros")
        
        # Output scaling
        scaled_outputs = {}
        for feature in self.output_features:
            if feature in data.columns:
                # Behandle leere/NaN-Werte
                feature_data = data[feature].fillna(0.0)
                if feature_data.dtype == 'object':
                    # Versuche zu konvertieren, falls es Strings sind
                    feature_data = pd.to_numeric(feature_data, errors='coerce').fillna(0.0)
                
                # DEBUG: Output scaling analysis
                print(f"[DEBUG] Output feature '{feature}' before scaling:")
                print(f"[DEBUG]   min: {feature_data.min()}, max: {feature_data.max()}")
                print(f"[DEBUG]   mean: {feature_data.mean()}, std: {feature_data.std()}")
                print(f"[DEBUG]   unique values: {feature_data.nunique()}")
                print(f"[DEBUG]   zero count: {len(feature_data[feature_data == 0])}")
                print(f"[DEBUG]   non-zero count: {len(feature_data[feature_data != 0])}")
                
                scaled_outputs[feature] = self.output_scalers[feature].transform(
                    feature_data.to_numpy().reshape(-1, 1)
                )
                
                # DEBUG: Output scaling results
                print(f"[DEBUG] Output feature '{feature}' after scaling:")
                print(f"[DEBUG]   min: {scaled_outputs[feature].min()}, max: {scaled_outputs[feature].max()}")
                print(f"[DEBUG]   mean: {scaled_outputs[feature].mean()}, std: {scaled_outputs[feature].std()}")
            else:
                scaled_outputs[feature] = np.zeros((len(data), 1))
                print(f"[WARN] Output feature '{feature}' not found, using zeros")
        
        # Create historical features tensor
        X_historical = torch.FloatTensor(np.column_stack([scaled_features[f] for f in self.features]))
        y = torch.FloatTensor(np.column_stack([scaled_outputs[f] for f in self.output_features]))
        
        # Simple approach: If weather_forecast_data is provided, use it for weather-informed mode
        print(f"[DEBUG] Weather forecast data provided: {weather_forecast_data is not None}")
        
        if weather_forecast_data is not None:
            print("[INFO] Weather-Informed mode: using forecast data for weather features")
            # Create weather-informed features by combining historical and forecast data
            X_weather_informed = self._create_weather_informed_features(data, weather_forecast_data, scaled_features, X_historical)
            X = X_weather_informed
        else:
            print("[INFO] Historical-only mode: using only historical data")
            X = X_historical
        
        # DEBUG: Final output analysis
        print(f"[DEBUG] ===== FINAL PREPROCESSOR OUTPUT =====")
        print(f"[DEBUG] X shape: {X.shape}, y shape: {y.shape}")
        print(f"[DEBUG] X min/max: {X.min():.8f}/{X.max():.8f}")
        print(f"[DEBUG] X mean/std: {X.mean():.8f}/{X.std():.8f}")
        print(f"[DEBUG] y min/max: {y.min():.8f}/{y.max():.8f}")
        print(f"[DEBUG] y mean/std: {y.mean():.8f}/{y.std():.8f}")
        print(f"[DEBUG] X first 3 rows:")
        print(X[:3])
        print(f"[DEBUG] y first 3 rows:")
        print(y[:3])
        
        return X, y
    
    def _create_weather_informed_features(self, historical_data: pd.DataFrame, 
                                        weather_forecast_data: pd.DataFrame, 
                                        scaled_historical_features: Dict,
                                        X_historical: torch.Tensor) -> torch.Tensor:
        """
        Create weather-informed features by combining historical and forecast data.
        
        Args:
            historical_data: Historical data DataFrame
            weather_forecast_data: Weather forecast data DataFrame
            scaled_historical_features: Pre-scaled historical features
            
        Returns:
            Combined features tensor with historical + forecast data
        """
        print(f"[DEBUG] Creating weather-informed features...")
        print(f"[DEBUG] Historical data shape: {historical_data.shape}")
        print(f"[DEBUG] Weather forecast data shape: {weather_forecast_data.shape}")
        
        # Process weather forecast data
        weather_forecast_data.columns = weather_forecast_data.columns.str.strip()
        weather_forecast_data = self._map_columns(weather_forecast_data)
        weather_forecast_data = self._process_datetime_features(weather_forecast_data)
        
        # Scale weather forecast features using the same scalers as historical data
        scaled_forecast_features = {}
        for feature in self.features:
            if feature in weather_forecast_data.columns:
                # Behandle leere/NaN-Werte
                feature_data = weather_forecast_data[feature].fillna(0.0)
                if feature_data.dtype == 'object':
                    # Versuche zu konvertieren, falls es Strings sind
                    feature_data = pd.to_numeric(feature_data, errors='coerce').fillna(0.0)
                
                scaled_forecast_features[feature] = self.feature_scalers[feature].transform(
                    feature_data.to_numpy().reshape(-1, 1)
                )
            else:
                scaled_forecast_features[feature] = np.zeros((len(weather_forecast_data), 1))
        
        # Create forecast features tensor
        X_forecast = torch.FloatTensor(np.column_stack([scaled_forecast_features[f] for f in self.features]))
        
        # Combine historical and forecast features
        # For each historical timestep, append the corresponding forecast timesteps
        combined_features = []
        
        print(f"[DEBUG] Loop range: range({len(historical_data)} - {self.sequence_length} - {self.forecast_steps}) = range({len(historical_data) - self.sequence_length - self.forecast_steps})")
        print(f"[DEBUG] X_forecast shape: {X_forecast.shape}")
        print(f"[DEBUG] self.sequence_length: {self.sequence_length}")
        print(f"[DEBUG] self.forecast_steps: {self.forecast_steps}")
        print(f"[DEBUG] self.settings: {self.settings}")
        
        for i in range(len(historical_data) - self.sequence_length - self.forecast_steps):
            # Historical features: 8 consecutive timesteps starting from i
            hist_start = i
            hist_end = hist_start + self.sequence_length
            hist_features = X_historical[hist_start:hist_end]  # Shape: (8, features)
            
            # Forecast features: 6 consecutive timesteps starting from hist_end
            forecast_start = hist_end
            forecast_end = forecast_start + self.forecast_steps
            
            if i < 3:  # Debug first 3 iterations
                print(f"[DEBUG] Iteration {i}: forecast_start={forecast_start}, forecast_end={forecast_end}, len(weather_forecast_data)={len(weather_forecast_data)}")
                print(f"[DEBUG] Condition: {forecast_end} <= {len(weather_forecast_data)} = {forecast_end <= len(weather_forecast_data)}")
            
            if forecast_end <= len(weather_forecast_data):
                # We have enough forecast data
                forecast_features = X_forecast[forecast_start:forecast_end]  # Shape: (forecast_steps, features)
                
                if i < 3:  # Debug first 3 iterations
                    print(f"[DEBUG] forecast_features shape: {forecast_features.shape}")
                    print(f"[DEBUG] hist_features shape: {hist_features.shape}")
                
                # Combine: [historical, forecast]
                combined_sequence = torch.cat([hist_features, forecast_features], dim=0)  # Shape: (sequence_length + forecast_steps, features)
                combined_features.append(combined_sequence)
                
                if i < 3:  # Debug first 3 iterations
                    print(f"[DEBUG] combined_sequence shape: {combined_sequence.shape}")
            else:
                # Not enough forecast data, skip this timestep
                if i < 3:  # Debug first 3 iterations
                    print(f"[DEBUG] Skipping iteration {i}: not enough forecast data")
                continue
        
        if combined_features:
            X_combined = torch.stack(combined_features)  # Shape: (n_sequences, 1 + forecast_steps, features)
            print(f"[DEBUG] Weather-informed features shape: {X_combined.shape}")
            return X_combined
        else:
            print(f"[WARN] No valid weather-informed sequences created, falling back to historical-only")
            return X_historical
    
    def transform(self, data: pd.DataFrame, weather_forecast_data: pd.DataFrame = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transform data for Multi-Step training.
        
        Args:
            data: Historical data DataFrame
            weather_forecast_data: Optional weather forecast data for Weather-Informed mode
            
        Returns:
            Tuple of (X, y) tensors
        """
        # Clean column names (remove trailing spaces)
        data.columns = data.columns.str.strip()
        
        data = self._map_columns(data)
        data = self._process_datetime_features(data)
        
        # Feature scaling
        scaled_features = {}
        for feature in self.features:
            if feature in data.columns:
                # Behandle leere/NaN-Werte
                feature_data = data[feature].fillna(0.0)
                if feature_data.dtype == 'object':
                    # Versuche zu konvertieren, falls es Strings sind
                    feature_data = pd.to_numeric(feature_data, errors='coerce').fillna(0.0)
                
                scaled_features[feature] = self.feature_scalers[feature].transform(
                    feature_data.to_numpy().reshape(-1, 1)
                )
            else:
                scaled_features[feature] = np.zeros((len(data), 1))
        
        # Output scaling
        scaled_outputs = {}
        for feature in self.output_features:
            if feature in data.columns:
                # Behandle leere/NaN-Werte
                feature_data = data[feature].fillna(0.0)
                if feature_data.dtype == 'object':
                    # Versuche zu konvertieren, falls es Strings sind
                    feature_data = pd.to_numeric(feature_data, errors='coerce').fillna(0.0)
                
                scaled_outputs[feature] = self.output_scalers[feature].transform(
                    feature_data.to_numpy().reshape(-1, 1)
                )
            else:
                scaled_outputs[feature] = np.zeros((len(data), 1))
        
        # Create features tensor
        X_historical = torch.FloatTensor(np.column_stack([scaled_features[f] for f in self.features]))
        y = torch.FloatTensor(np.column_stack([scaled_outputs[f] for f in self.output_features]))
        
        # Simple approach: If weather_forecast_data is provided, use it for weather-informed mode
        print(f"[DEBUG] Weather forecast data provided: {weather_forecast_data is not None}")
        
        if weather_forecast_data is not None:
            print("[INFO] Weather-Informed mode: using forecast data for weather features")
            # Create weather-informed features by combining historical and forecast data
            X_weather_informed = self._create_weather_informed_features(data, weather_forecast_data, scaled_features, X_historical)
            X = X_weather_informed
        else:
            print("[INFO] Historical-only mode: using only historical data")
            X = X_historical
        
        return X, y
    
    def inverse_transform_output(self, scaled_output: torch.Tensor, context_data: Optional[Dict] = None) -> np.ndarray:
        """
        Transformiere skalierte Ausgaben zurück in den Originalbereich.
        
        Args:
            scaled_output: Skalierte Modellausgaben
            context_data: Optional dictionary with context data for constraints (e.g., {'GS_10': array, 'hour': array})
        """
        # Store context data for physical constraints
        if context_data is not None:
            if 'GS_10' in context_data:
                self.current_gs10 = context_data['GS_10']
            if 'hour' in context_data:
                self.current_hours = context_data['hour']
        else:
            self.current_gs10 = None
            self.current_hours = None
        
        scaled_output_np = scaled_output.numpy()
        original_outputs = {}
        
        for i, feature in enumerate(self.output_features):
            # Check if scaler is fitted
            if not hasattr(self.output_scalers[feature], 'scale_') or self.output_scalers[feature].scale_ is None:
                print(f"[ERROR] Scaler for '{feature}' is not fitted!")
                print(f"[ERROR] Scaler attributes: {[attr for attr in dir(self.output_scalers[feature]) if not attr.startswith('_')]}")
                raise ValueError(f"Scaler for {feature} is not fitted")
            
            # Extract the feature data
            feature_data = scaled_output_np[:, i:i+1]
            
            # Perform inverse transform
            original_outputs[feature] = self.output_scalers[feature].inverse_transform(feature_data)
        
        result = np.column_stack([original_outputs[f] for f in self.output_features])
        
        return result
    
    def _apply_physical_constraints(self, output_array: np.ndarray) -> np.ndarray:
        """
        SCIENTIFIC APPROACH: Return raw model predictions without any constraints.
        This allows evaluation of the model's true learning capability and physics understanding.
        
        Args:
            output_array: Model output array with shape (n_samples, n_outputs)
            
        Returns:
            Raw output array (no constraints applied)
        """
        # Return raw predictions without any modifications
        return output_array
    
    def create_sequences(self, X: torch.Tensor, y: torch.Tensor, sequence_length: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create sequences for Multi-Step LSTM training.
        
        Args:
            X: Input features tensor
            y: Target values tensor  
            sequence_length: Length of input sequences
            
        Note:
            Multi-Step: Input[t:t+seq_len] -> Target[t+seq_len:t+seq_len+forecast_steps]
        """
        if sequence_length is None:
            sequence_length = self.sequence_length
        
        # print(f"[DEBUG] ===== MULTI-STEP SEQUENCE CREATION DEBUG =====")
        # print(f"[DEBUG] Input X shape: {X.shape}")
        # print(f"[DEBUG] Input y shape: {y.shape}")
        # print(f"[DEBUG] Sequence length: {sequence_length}")
        # print(f"[DEBUG] Forecast steps: {self.forecast_steps}")
        # print(f"[DEBUG] Multi-Step mode: Input[t:t+{sequence_length}] -> Target[t+{sequence_length}:t+{sequence_length}+{self.forecast_steps}]")
        
        sequences = []
        targets = []
        
        for i in range(len(X) - sequence_length - self.forecast_steps + 1):
            # Input sequence: from i to i+sequence_length
            sequences.append(X[i:i+sequence_length])
            
            # Multi-step: targets are the next forecast_steps timestamps
            target_start = i + sequence_length
            target_end = target_start + self.forecast_steps
            
            if target_end <= len(y):
                # Shape: (forecast_steps, output_features)
                target_sequence = y[target_start:target_end]
                targets.append(target_sequence)
            else:
                # Skip this sequence if targets are out of bounds
                continue
        
        X_seq = torch.stack(sequences)
        y_seq = torch.stack(targets)
        
        # print(f"[DEBUG] Output X_seq shape: {X_seq.shape}")
        # print(f"[DEBUG] Output y_seq shape: {y_seq.shape}")
        # print(f"[DEBUG] Multi-step targets shape: (batch_size, {self.forecast_steps}, {y.shape[1]})")
        # print(f"[DEBUG] ===== END MULTI-STEP SEQUENCE CREATION DEBUG =====")
        
        return X_seq, y_seq

"""
Note: Resolution handling has been moved upstream. The training now receives
weather-integrated CSVs already in the correct resolution (5min/10min/1h).
This module no longer performs resampling of integrated data.
"""