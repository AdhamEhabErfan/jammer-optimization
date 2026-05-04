import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HybridJammerNet(nn.Module):
    """
    Hybrid network: LSTM for prediction + Power allocation head.
    Outputs power distribution across all bands.
    """
    
    def __init__(self, num_bands, sensing_window, hidden_size=128, num_layers=2):
        super().__init__()
        self.num_bands = num_bands
        
        # Sequence encoder
        self.lstm = nn.LSTM(
            input_size=num_bands,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        # Prediction head: which band will TX use
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_bands)
        )
        
        # Power allocation head: how to distribute power
        self.power_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_bands)
        )
    
    def forward(self, x):
        lstm_out, (h_n, _) = self.lstm(x)
        features = h_n[-1]  # Last layer's hidden state
        
        # Predict next band probabilities
        pred_logits = self.prediction_head(features)
        pred_probs = F.softmax(pred_logits, dim=-1)
        
        # Power allocation (softmax ensures sum = 1)
        power_logits = self.power_head(features)
        power_alloc = F.softmax(power_logits, dim=-1)
        
        return pred_logits, pred_probs, power_alloc