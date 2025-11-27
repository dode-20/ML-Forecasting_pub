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

# Dynamic imports that work in both Docker and offline environments
def import_lstm_modules():
    """Import LSTM modules dynamically based on environment"""
    try:
        # Try relative imports first (Docker environment)
        from .model import LSTMModel
        from .preprocess import DataPreprocessor
        return LSTMModel, DataPreprocessor
    except ImportError:
        try:
            # Try absolute imports (offline environment)
            from model import LSTMModel
            from preprocess import DataPreprocessor
            return LSTMModel, DataPreprocessor
        except ImportError:
            # Fallback: try to add the current directory to path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)
            from model import LSTMModel
            from preprocess import DataPreprocessor
            return LSTMModel, DataPreprocessor

# Import the modules
LSTMModel, DataPreprocessor = import_lstm_modules()

class LSTMTrainer:
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
            dropout=dropout
        ).to(self.device)
        
    def train(self, data: pd.DataFrame) -> Dict[str, List[float]]:
        """Train the LSTM model with the given data and settings."""
        print(f"Training running on: {self.device} in {self.mode} mode")
        
        # DEBUG: Input data analysis
        print(f"[DEBUG] ===== TRAINER INPUT DATA ANALYSIS =====")
        print(f"[DEBUG] Input data shape: {data.shape}")
        print(f"[DEBUG] Input data columns: {list(data.columns)}")
        print(f"[DEBUG] Input data first 3 rows:")
        print(data.head(3))
        print(f"[DEBUG] Input data statistics:")
        print(data.describe())
        
        # Initialize preprocessor
        self.preprocessor = DataPreprocessor(
            features=self.settings["features"],
            output_features=self.settings["output"],
            sequence_length=self.sequence_length
        )
        
        # IMPORTANT: Do not resample here. The provided dataframe must already
        # be created from the matching resolution CSV (5min/10min/1h).

        print("Preprocessing data...")
        # Preprocess data
        X, y = self.preprocessor.fit_transform(data)
        
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
        input_size = X.shape[1]  # Number of features after preprocessing
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
        # Create sequences with target_shift=0 for same-time prediction (DWD compatibility)
        # target_shift=0: Input[t:t+seq_len] -> Target[t] (same time)
        # FIXED: Target is now ALWAYS the last timestamp of the input sequence
        # This means: Input[t:t+seq_len] -> Target[t+seq_len-1] (last element)
        
        print(f"[INFO] Using corrected sequence creation logic")
        print(f"[INFO] This means: Wetterdaten[t:t+seq_len] -> PV-Leistung[t+seq_len-1] (letzter Zeitstempel)")
        print(f"[INFO] Example: Sequenz[11:00,12:00,13:00,14:00] -> PV-Vorhersage für 14:00")
        
        # Create sequences
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
        early_stopping_patience = 15  # Increased from 5 to 15
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