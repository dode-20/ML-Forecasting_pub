import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2, forecast_steps=1):
        super(LSTMModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.forecast_steps = forecast_steps
        self.dropout = dropout
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Multi-step output layer: hidden_size -> (forecast_steps * output_size)
        self.fc = nn.Linear(hidden_size, forecast_steps * output_size)
        
        # ReLU activation to prevent negative predictions (physically impossible for PV)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        batch_size = x.size(0)
        
        # Initialize hidden states
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # Multi-step prediction: use last timestep for all forecast steps
        # Shape: (batch_size, forecast_steps * output_size)
        out = self.fc(out[:, -1, :])
        
        # Apply ReLU to prevent negative predictions (physically impossible for PV)
        out = self.relu(out)
        
        # Reshape to (batch_size, forecast_steps, output_size)
        out = out.view(batch_size, self.forecast_steps, self.output_size)
        
        return out