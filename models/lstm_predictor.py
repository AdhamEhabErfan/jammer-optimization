import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMPredictor(nn.Module):
    """
    LSTM network that predicts the next frequency band(s) 
    based on historical observations.
    """
    
    def __init__(self, num_bands, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.num_bands = num_bands
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(
            input_size=num_bands,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Attention mechanism for important time steps
        self.attention = nn.Linear(hidden_size, 1)
        
        # Output: probability distribution over bands
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, num_bands)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: (batch, seq_len, num_bands)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Attention weights
        attn_weights = F.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        
        # Predict probability distribution
        x = F.relu(self.fc1(context))
        x = self.dropout(x)
        logits = self.fc2(x)
        
        return logits  # (batch, num_bands)
    
    def predict_proba(self, x):
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)
    
    def predict_top_k(self, x, k=3):
        """Return top-k most likely bands"""
        probs = self.predict_proba(x)
        top_probs, top_indices = torch.topk(probs, k, dim=-1)
        return top_indices, top_probs