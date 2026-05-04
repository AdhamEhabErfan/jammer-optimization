import torch
import os
import numpy as np
from config import Config
from models.lstm_predictor import LSTMPredictor
from models.dqn_agent import DQN
from models.hybrid_model import HybridJammerNet

config = Config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

# Track which models loaded
loaded = {'lstm': False, 'dqn': False, 'hybrid': False}

# 1. LSTM
try:
    lstm = LSTMPredictor(config.NUM_BANDS, config.LSTM_HIDDEN_SIZE,
                         config.LSTM_NUM_LAYERS, config.DROPOUT).to(device)
    lstm.load_state_dict(torch.load('lstm_predictor.pth', map_location=device))
    lstm.eval()
    loaded['lstm'] = True
    print("✅ LSTM loaded")
except Exception as e:
    print(f"❌ LSTM failed: {e}\n")

# 2. DQN
try:
    dqn = DQN(config.NUM_BANDS, config.SENSING_WINDOW).to(device)
    dqn.load_state_dict(torch.load('dqn_jammer.pth', map_location=device))
    dqn.eval()
    loaded['dqn'] = True
    print("✅ DQN loaded")
except Exception as e:
    print(f"❌ DQN failed: {e}\n")

# 3. Hybrid
try:
    hybrid = HybridJammerNet(config.NUM_BANDS, config.SENSING_WINDOW,
                             config.LSTM_HIDDEN_SIZE).to(device)
    hybrid.load_state_dict(torch.load('hybrid_jammer.pth', map_location=device))
    hybrid.eval()
    loaded['hybrid'] = True
    print("✅ Hybrid loaded")
except Exception as e:
    print(f"❌ Hybrid failed: {e}\n")

# Final verdict
print("\n" + "=" * 50)
if all(loaded.values()):
    print("🎉 SUCCESS — all 3 models loaded!")
else:
    failed = [k for k, v in loaded.items() if not v]
    print(f"⚠️  FAILED: {failed}")
    print("👉 Fix: update LSTM_HIDDEN_SIZE in config.py to match training")
print("=" * 50)