import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import time
import sys
from datetime import datetime

# Import modules using absolute paths to avoid conflicts
import importlib.util

# Import LSTMModel from current directory
model_path = Path(__file__).parent / "model.py"
spec = importlib.util.spec_from_file_location("multistep_model", model_path)
model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_module)
LSTMModel = model_module.LSTMModel

# Import DataPreprocessor from current directory  
preprocess_path = Path(__file__).parent / "preprocess.py"
spec = importlib.util.spec_from_file_location("multistep_preprocess", preprocess_path)
preprocess_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preprocess_module)
DataPreprocessor = preprocess_module.DataPreprocessor

class MultiStepLSTMTrainer:
    def __init__(self, settings: Dict[str, Any], device: str = "auto", external_timestamp: Optional[str] = None, external_model_dir: Optional[Path] = None):
        self.settings = settings
        self.data_resolution: str = str(settings.get("data_resolution", "5min"))
        
        # Device selection logic
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device == "cuda":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                print("Warning: CUDA requested but not available. Falling back to CPU.")
                self.device = torch.device("cpu")
        elif device == "cpu":
            self.device = torch.device("cpu")
        else:
            print(f"Warning: Unknown device '{device}'. Falling back to auto detection.")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model: Optional[LSTMModel] = None
        self.preprocessor: Optional[DataPreprocessor] = None
        self.mode = self._detect_environment_mode()
        
        # NEU: Verwende externen Timestamp und Ordner falls vorhanden, sonst erstelle neue
        if external_timestamp and external_model_dir:
            # Verwende externe Werte (z.B. vom Test-Skript)
            self.timestamp = external_timestamp
            self.model_dir = external_model_dir
            print(f"Using external timestamp: {self.timestamp}")
            print(f"Using external model directory: {self.model_dir}")
        else:
            # Fallback: Create own values (for direct use of trainer)
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if Path("/app/results").exists():
                self.model_dir = Path("/app/results/trained_models/lstm") / f"{self.timestamp}_lstm"
            else:
                self.model_dir = Path("results/trained_models/lstm") / f"{self.timestamp}_lstm"
            print(f"Created new timestamp: {self.timestamp}")
            print(f"Created new model directory: {self.model_dir}")
        
        # Model name wird immer aus den Settings genommen
        self.model_name = self.settings.get("model_name", "lstm_model")
        
        # Stelle sicher, dass der Ordner existiert
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.sequence_length = int(self.settings.get("sequence_length", 864))
        
        # Parse forecast mode configuration
        forecast_mode = self.settings.get("forecast_mode", {"mode": "one-step", "forecast_steps": 1})
        self.forecast_mode = forecast_mode.get("mode", "one-step")
        self.forecast_steps = int(forecast_mode.get("forecast_steps", 1))
        
        # Calculate forecast horizon from steps and resolution
        self.forecast_horizon_hours = self._calculate_forecast_horizon()
        
        # Calculate and display sequence-to-forecast ratio
        self._display_sequence_to_forecast_ratio()
        # Resolution manager
        # Resolution handling is now done upstream by selecting the correct
        # weather-integrated CSV (5min/10min/1h). No resampling here.
        
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
    
    def _calculate_forecast_horizon(self) -> float:
        """Calculate forecast horizon in hours from steps and data resolution"""
        resolution = self.data_resolution.lower()
        
        # Convert resolution to hours
        if resolution == "5min":
            resolution_hours = 5 / 60
        elif resolution == "10min":
            resolution_hours = 10 / 60
        elif resolution == "15min":
            resolution_hours = 15 / 60
        elif resolution == "30min":
            resolution_hours = 30 / 60
        elif resolution == "1h":
            resolution_hours = 1.0
        elif resolution == "2h":
            resolution_hours = 2.0
        else:
            # Default to 1h if unknown resolution
            print(f"[WARN] Unknown resolution '{resolution}', defaulting to 1h")
            resolution_hours = 1.0
        
        horizon_hours = self.forecast_steps * resolution_hours
        print(f"[INFO] Calculated forecast horizon: {self.forecast_steps} steps × {resolution} = {horizon_hours:.2f} hours")
        
        return horizon_hours
    
    def _display_sequence_to_forecast_ratio(self):
        """Calculate and display sequence-to-forecast ratio with scientific interpretation"""
        sequence_length = self.sequence_length
        sequence_to_forecast_ratio = sequence_length / self.forecast_steps
        
        print(f"[INFO] ===== SEQUENCE-TO-FORECAST RATIO ANALYSIS =====")
        print(f"[INFO] Sequence length: {sequence_length}")
        print(f"[INFO] Forecast steps: {self.forecast_steps}")
        print(f"[INFO] Sequence-to-Forecast Ratio: {sequence_to_forecast_ratio:.2f}")
        print(f"[INFO] Input-Output Ratio: {sequence_to_forecast_ratio:.2f}")
        
        # Scientific interpretation
        if sequence_to_forecast_ratio > 1.0:
            print(f"[INFO] → Model has more input context than forecast horizon (conservative)")
            print(f"[INFO] → Advantage: Rich historical context for predictions")
            print(f"[INFO] → Risk: Potential overfitting to historical patterns")
        elif sequence_to_forecast_ratio == 1.0:
            print(f"[INFO] → Model has equal input context and forecast horizon (balanced)")
            print(f"[INFO] → Advantage: Optimal balance between context and generalization")
            print(f"[INFO] → Risk: Minimal - well-balanced approach")
        else:
            print(f"[INFO] → Model has less input context than forecast horizon (aggressive)")
            print(f"[INFO] → Advantage: Efficient learning, good generalization")
            print(f"[INFO] → Risk: Potential underfitting due to limited context")
        
        print(f"[INFO] ===== END SEQUENCE-TO-FORECAST RATIO ANALYSIS =====")
        
    def setup_model(self, input_size: int, output_size: int) -> None:
        """Initialisiere das LSTM-Modell mit den angegebenen Parametern."""
        # Optimized architecture for better performance
        hidden_size = self.settings.get("hidden_size", 128)  # Increased from 64 to 128
        num_layers = self.settings.get("num_layers", 3)      # Increased from 2 to 3
        dropout = self.settings.get("dropout", 0.3)          # Increased from 0.2 to 0.3
        
        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            dropout=dropout,
            forecast_steps=self.forecast_steps
        ).to(self.device)
        
    def train(self, data: pd.DataFrame, weather_forecast_data: pd.DataFrame = None) -> Dict[str, List[float]]:
        """Train the Weather-Informed Multi-Step LSTM model with the given data and settings."""
        print(f"Training running on: {self.device} in {self.mode} mode")
        
        # Check if weather-informed mode should be enabled based on settings
        weather_enabled = self.settings.get("weather_data", {}).get("use_weatherData", False)
        weather_features = self.settings.get("weather_data", {}).get("weather_features", [])
        weather_informed = weather_enabled and weather_features and weather_forecast_data is not None
        
        if weather_enabled and weather_forecast_data is None:
            print(f"[WARN] Weather data enabled but no weather forecast data provided!")
            print(f"[WARN] Falling back to historical-only mode")
        elif weather_enabled and not weather_features:
            print(f"[WARN] Weather data enabled but no weather features configured!")
            print(f"[WARN] Falling back to historical-only mode")
        
        if weather_informed:
            print(f"[INFO] Weather-informed multi-step mode: {self.forecast_steps} forecast steps")
        else:
            print(f"[INFO] Multi-step mode: {self.forecast_steps} forecast steps")
        
        # DEBUG: Input data analysis
        print(f"[DEBUG] ===== TRAINER INPUT DATA ANALYSIS =====")
        print(f"[DEBUG] Historical data shape: {data.shape}")
        print(f"[DEBUG] Historical data columns: {list(data.columns)}")
        print(f"[DEBUG] Historical data first 3 rows:")
        print(data.head(3))
        print(f"[DEBUG] Historical data statistics:")
        print(data.describe())
        
        # Initialize preprocessor
        self.preprocessor = DataPreprocessor(
            features=self.settings["features"],
            output_features=self.settings["output"],
            sequence_length=self.sequence_length,
            forecast_steps=self.forecast_steps,
            settings=self.settings
        )
        
        # IMPORTANT: Do not resample here. The provided dataframe must already
        # be created from the matching resolution CSV (5min/10min/1h).

        print("Preprocessing data...")
        # Preprocess data (with weather forecast if available)
        X, y = self.preprocessor.fit_transform(data, weather_forecast_data, self.settings)
        
        # DEBUG: Preprocessed data analysis
        print(f"[DEBUG] ===== PREPROCESSED DATA ANALYSIS =====")
        print(f"[DEBUG] X shape: {X.shape}, y shape: {y.shape}")
        print(f"[DEBUG] X min/max: {X.min():.8f}/{X.max():.8f}")
        print(f"[DEBUG] X mean/std: {X.mean():.8f}/{X.std():.8f}")
        print(f"[DEBUG] y min/max: {y.min():.8f}/{y.max():.8f}")
        print(f"[DEBUG] y mean/std: {y.mean():.8f}/{y.std():.8f}")
        print(f"[DEBUG] X first 3 rows:")
        print(X[:3])
        print(f"[DEBUG] y first 3 rows:")
        print(y[:3])
        
        # Setup model with correct input size after preprocessing
        # For weather-informed models, X has shape (n_sequences, sequence_length + forecast_steps, features)
        if len(X.shape) == 3:
            # Weather-informed mode: (n_sequences, sequence_length + forecast_steps, features)
            input_size = X.shape[2]  # Number of features
            sequence_length = X.shape[1]  # Total sequence length (historical + forecast)
            print(f"[DEBUG] Weather-informed mode detected: X shape {X.shape}")
            print(f"[DEBUG] Input size: {input_size}, Sequence length: {sequence_length}")
        else:
            # Historical-only mode: (n_samples, features)
            input_size = X.shape[1]  # Number of features
            sequence_length = self.sequence_length
            print(f"[DEBUG] Historical-only mode detected: X shape {X.shape}")
            print(f"[DEBUG] Input size: {input_size}, Sequence length: {sequence_length}")
        
        output_size = y.shape[1]  # Number of output features
        
        # DEBUG: Model setup analysis
        print(f"[DEBUG] ===== MODEL SETUP ANALYSIS =====")
        print(f"[DEBUG] Input size: {input_size}")
        print(f"[DEBUG] Output size: {output_size}")
        print(f"[DEBUG] Expected features: {self.settings['features']}")
        print(f"[DEBUG] Expected outputs: {self.settings['output']}")
        
        self.setup_model(input_size, output_size)
        
        # DEBUG: Model architecture analysis
        print(f"[DEBUG] ===== MODEL ARCHITECTURE ANALYSIS =====")
        print(f"[DEBUG] Model type: {type(self.model)}")
        print(f"[DEBUG] Model parameters: {sum(p.numel() for p in self.model.parameters())}")
        print(f"[DEBUG] Model device: {next(self.model.parameters()).device}")
        
        print("Creating sequences...")
        # Multi-Step sequence creation: Input[t:t+seq_len] -> Target[t+seq_len:t+seq_len+forecast_steps]
        # This means: For each input sequence, predict the next forecast_steps timestamps
        
        print(f"[INFO] Forecast mode: {self.forecast_mode}")
        print(f"[INFO] Forecast steps: {self.forecast_steps}")
        print(f"[INFO] Forecast horizon: {self.forecast_horizon_hours} hours")
        
        print(f"[INFO] Multi-Step mode: Input[t:t+seq_len] -> Target[t+seq_len:t+seq_len+{self.forecast_steps}]")
        print(f"[INFO] Example: Sequenz[11:00,12:00,13:00,14:00] -> PV-Vorhersage für 15:00,16:00,17:00,18:00,19:00,20:00")
        
        # Create sequences
        print(f"[INFO] Creating Weather-Informed Multi-Step sequences...")
        if len(X.shape) == 3:
            # Weather-informed mode: X is already in sequence format
            print(f"[INFO] Weather-informed mode: X already in sequence format")
            X_seq = X
            # Create multi-step targets for weather-informed mode
            y_seq = self._create_multistep_targets(y, len(X_seq))
        else:
            # Historical-only mode: create sequences
            print(f"[INFO] Historical-only mode: creating sequences")
            X_seq, y_seq = self.preprocessor.create_sequences(X, y)
        
        # DEBUG: Sequence data analysis
        print(f"[DEBUG] ===== SEQUENCE DATA ANALYSIS =====")
        print(f"[DEBUG] X_seq shape: {X_seq.shape}")
        print(f"[DEBUG] y_seq shape: {y_seq.shape}")
        print(f"[DEBUG] X_seq min/max: {X_seq.min():.8f}/{X_seq.max():.8f}")
        print(f"[DEBUG] y_seq min/max: {y_seq.min():.8f}/{y_seq.max():.8f}")
        
        # Interval split for validation
        val_split = float(self.settings.get('validation_split', 0.15))
        if self.settings.get("use_validation_set") == "Yes":
            if val_split > 0:
                n = round(1 / val_split)
            else:
                n = len(X_seq) + 1  # Falls 0, alles Training
            idx = np.arange(len(X_seq))
            val_mask = (idx % n) == 0
            train_mask = ~val_mask
            X_train, X_val = X_seq[train_mask], X_seq[val_mask]
            y_train, y_val = y_seq[train_mask], y_seq[val_mask]
            train_dataset = TensorDataset(X_train, y_train)
            val_dataset = TensorDataset(X_val, y_val)
            train_loader = DataLoader(
                train_dataset, 
                batch_size=int(self.settings["batch_size"]), 
                shuffle=bool(self.settings["shuffle"])
            )
            val_loader = DataLoader(val_dataset, batch_size=int(self.settings["batch_size"]))
        else:
            train_dataset = TensorDataset(X_seq, y_seq)
            train_loader = DataLoader(
                train_dataset, 
                batch_size=int(self.settings["batch_size"]), 
                shuffle=bool(self.settings["shuffle"])
            )
            val_loader = None
        
        # Optimizer and loss function
        optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.settings["learning_rate"]))
        loss_fn = self._get_loss_function(str(self.settings["loss_function"]))
        
        # Training
        best_val_loss = float('inf')
        early_stopping_counter = 0
        early_stopping_patience = self.settings.get("early_stopping_patience", 15)  # Use config value or default to 15
        early_stopping_min_delta = 0.0001  # Minimum improvement for early stopping
        training_history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [] if val_loader else []
        }
        
        # Create progress bar for epochs
        epochs_pbar = tqdm(range(int(self.settings["epochs"])), desc="Training", position=0, file=sys.stderr)
        
        for epoch in epochs_pbar:
            self.model.train()
            train_loss = 0.0
            
            # Training loop
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
                # Calculate progress
                batch_progress = (batch_idx + 1) / len(train_loader)
                epoch_progress = (epoch + batch_progress) / int(self.settings["epochs"])
                
                # Update training status (only in Docker mode)
                if self.mode == "docker":
                    try:
                        from ..main import update_training_status
                        update_training_status(
                            self.settings["model_name"],
                            epoch + 1,
                            loss.item(),
                            epoch_progress * 100,
                            progress_info=f"Epoch {epoch + 1}/{self.settings['epochs']} - Loss: {loss.item():.4f}"
                        )
                    except ImportError:
                        # Fallback if update_training_status is not available
                        pass
            
            # Calculate average training loss
            train_loss /= len(train_loader)
            training_history["train_loss"].append(train_loss)
            
            # Update progress bar
            epochs_pbar.set_postfix({
                "loss": f"{train_loss:.4f}",
                "epoch": f"{epoch + 1}/{self.settings['epochs']}"
            })
            
            # Validation
            if val_loader:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for data, target in val_loader:
                        data, target = data.to(self.device), target.to(self.device)
                        output = self.model(data)
                        val_loss += loss_fn(output, target).item()
                
                val_loss /= len(val_loader)
                training_history["val_loss"].append(val_loss)
                
                # Update training status with validation loss (only in Docker mode)
                if self.mode == "docker":
                    try:
                        from ..main import update_training_status
                        update_training_status(
                            self.settings["model_name"],
                            epoch + 1,
                            train_loss,
                            (epoch + 1) / int(self.settings["epochs"]) * 100,
                            val_loss,
                            progress_info=f"Epoch {epoch + 1}/{self.settings['epochs']} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                        )
                    except ImportError:
                        # Fallback if update_training_status is not available
                        pass
                
                # Early stopping
                if val_loss < (best_val_loss - early_stopping_min_delta):
                    best_val_loss = val_loss
                    early_stopping_counter = 0
                    self._save_model()
                    print("New best model saved!", file=sys.stderr)
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter >= early_stopping_patience:  # Early stopping after 15 epochs without improvement
                        print(f"Early stopping at Epoch {epoch+1}", file=sys.stderr)
                        break
            else:
                print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}", file=sys.stderr)
                self._save_model()
            
            # Small pause for status updates (only in Docker mode)
            if self.mode == "docker":
                time.sleep(0.1)
        
        print("\nTraining completed!", file=sys.stderr)
        return training_history
    
    def _create_multistep_targets(self, y: torch.Tensor, n_sequences: int) -> torch.Tensor:
        """
        Create multi-step targets for weather-informed mode.
        
        Args:
            y: Single-step targets tensor (n_samples, 1)
            n_sequences: Number of sequences to create
            
        Returns:
            Multi-step targets tensor (n_sequences, forecast_steps, 1)
        """
        # Create multi-step targets by taking consecutive forecast_steps
        y_multistep = []
        for i in range(n_sequences):
            # Take forecast_steps consecutive targets starting from position i + sequence_length
            # This ensures targets correspond to the forecast period after the historical input
            start_idx = i + self.settings.get("sequence_length", 8)
            end_idx = min(start_idx + self.forecast_steps, len(y))
            
            if end_idx - start_idx == self.forecast_steps:
                # We have enough targets
                target_sequence = y[start_idx:end_idx]  # Shape: (forecast_steps, 1)
                y_multistep.append(target_sequence)
            else:
                # Not enough targets, pad with zeros
                target_sequence = y[start_idx:end_idx]  # Shape: (available_steps, 1)
                padding_needed = self.forecast_steps - (end_idx - start_idx)
                if padding_needed > 0:
                    padding = torch.zeros(padding_needed, 1)
                    target_sequence = torch.cat([target_sequence, padding], dim=0)
                y_multistep.append(target_sequence)
        
        return torch.stack(y_multistep)  # Shape: (n_sequences, forecast_steps, 1)
    
    def _get_loss_function(self, loss_name: str) -> nn.Module:
        """Gebe die entsprechende Loss-Funktion zurück."""
        loss_functions: Dict[str, nn.Module] = {
            "MSE": nn.MSELoss(),
            "RMSE": lambda x, y: torch.sqrt(nn.MSELoss()(x, y)),
            "MAE": nn.L1Loss(),
            "MAPE": lambda x, y: torch.mean(torch.abs((y - x) / (y + 1e-8))) * 100
        }
        return loss_functions.get(loss_name, nn.MSELoss())
    
    def scaler_to_dict(self, scaler):
        params = scaler.get_params()
        # Add all learned attributes if present
        for attr in ["min_", "scale_", "data_min_", "data_max_", "data_range_", "n_samples_seen_"]:
            if hasattr(scaler, attr):
                val = getattr(scaler, attr)
                # Convert numpy arrays to lists for JSON
                if isinstance(val, (np.ndarray, list)):
                    params[attr] = [float(x) for x in val]
                else:
                    params[attr] = float(val) if isinstance(val, (np.floating, float, int)) else val
        return params
    
    def _save_model(self) -> None:
        """Speichere das Modell und die Preprocessing-Parameter."""
        if self.model is None or self.preprocessor is None:
            raise ValueError("Model or preprocessor not initialized")
        # --- Timestamp und Ordner werden jetzt aus Instanzvariablen genommen ---
        timestamp = self.timestamp
        model_name = self.model_name
        model_dir = self.model_dir
        # Save model weights
        torch.save(self.model.state_dict(), model_dir / f"{timestamp}_lstm.pth")
        # Save model configuration
        model_config = {
            "model_name": model_name,
            "model_type": self.settings.get("model_type", "LSTM"),
            "device": str(self.device),
            "input_size": self.model.input_size,
            "hidden_size": self.model.hidden_size,
            "num_layers": self.model.num_layers,
            "output_size": self.model.output_size,
            "dropout": self.model.dropout,
            "sequence_length": self.sequence_length,
            "data_resolution": self.data_resolution,
            "training_settings": {
                "features": self.settings["features"],
                "output": self.settings["output"],
                "batch_size": self.settings["batch_size"],
                "epochs": self.settings["epochs"],
                "learning_rate": self.settings["learning_rate"],
                "loss_function": self.settings["loss_function"],
                "shuffle": self.settings["shuffle"],
                "use_validation_set": self.settings.get("use_validation_set", "Yes"),
                "validation_split": self.settings.get("validation_split", 0.15),
                "sequence_length": self.sequence_length,
                "data_resolution": self.data_resolution,
                "target_shift": "N/A"  # No longer used - target is always last element of sequence
            }
        }
        with open(model_dir / f"model_config_{timestamp}_lstm.json", "w") as f:
            json.dump(model_config, f, indent=4)
        # Save preprocessor parameters
        preprocessor_params = {
            "features": self.settings["features"],
            "output_features": self.settings["output"],
            "feature_scalers": {f: self.scaler_to_dict(s) for f, s in self.preprocessor.feature_scalers.items()},
            "output_scalers": {f: self.scaler_to_dict(s) for f, s in self.preprocessor.output_scalers.items()}
        }
        with open(model_dir / f"preprocessor_{timestamp}_lstm.json", "w") as f:
            json.dump(preprocessor_params, f, indent=4)

def train_model(model, train_loader, val_loader, epochs, learning_rate, model_name, sse_manager, device="auto"):
    """Trains the model and sends updates via SSE."""
    # Device selection logic
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            device = torch.device("cpu")
    elif device == "cpu":
        device = torch.device("cpu")
    else:
        print(f"Warning: Unknown device '{device}'. Falling back to auto detection.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move model to device
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training History
    history = {
        "train_loss": [],
        "val_loss": []
    }
    
    # Create tqdm progress bar for stderr
    pbar = tqdm(range(epochs), file=sys.stderr, desc="Training Progress")
    
    for epoch in pbar:
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Move data to device
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            # Calculate batch progress
            batch_progress = (batch_idx + 1) / len(train_loader)
            epoch_progress = (epoch + batch_progress) / epochs
            
            # Send status update (only if sse_manager is provided)
            if sse_manager is not None:
                status = {
                    "current_epoch": epoch + 1,
                    "total_epochs": epochs,
                    "current_loss": loss.item(),
                    "epoch_progress": epoch_progress,
                    "status": "training"
                }
                sse_manager.send_update(model_name, status)
            
            # Small pause for updates
            time.sleep(0.1)
        
        # Validate after each epoch
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                # Move data to device
                data, target = data.to(device), target.to(device)
                
                output = model(data)
                val_loss += criterion(output, target).item()
        
        # Calculate average loss
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        # Save history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        
        # Update tqdm description
        pbar.set_description(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        
        # Send final update for this epoch (only if sse_manager is provided)
        if sse_manager is not None:
            status = {
                "current_epoch": epoch + 1,
                "total_epochs": epochs,
                "current_loss": train_loss,
                "validation_loss": val_loss,
                "epoch_progress": (epoch + 1) / epochs,
                "status": "training"
            }
            sse_manager.send_update(model_name, status)
    
    # Send completion update (only if sse_manager is provided)
    if sse_manager is not None:
        final_status = {
            "current_epoch": epochs,
            "total_epochs": epochs,
            "current_loss": train_loss,
            "validation_loss": val_loss,
            "epoch_progress": 1.0,
            "status": "completed"
        }
        sse_manager.send_update(model_name, final_status)
    
    return history