import torch
import numpy as np
from config import Config
from frequency_hopping import FrequencyHoppingTransmitter
from models.lstm_predictor import LSTMPredictor

config = Config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load LSTM
lstm = LSTMPredictor(config.NUM_BANDS, config.LSTM_HIDDEN_SIZE,
                     config.LSTM_NUM_LAYERS, config.DROPOUT).to(device)
lstm.load_state_dict(torch.load('lstm_predictor.pth', map_location=device))
lstm.eval()

# Generate FH sequence with the SAME seed used in training
print(f"Testing with FH_SEED={config.FH_SEED}, algorithm={config.FH_ALGORITHM}")
tx = FrequencyHoppingTransmitter(config.NUM_BANDS, config.FH_ALGORITHM, 
                                  seed=config.FH_SEED)
seq = tx.generate_sequence(2000)

# Test prediction accuracy
correct = 0
total = 0
top3_correct = 0

for i in range(len(seq) - config.SENSING_WINDOW - 1):
    window = seq[i:i + config.SENSING_WINDOW]
    target = seq[i + config.SENSING_WINDOW]
    onehot = np.eye(config.NUM_BANDS)[window].astype(np.float32)
    x = torch.from_numpy(onehot).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = lstm(x)
        pred = logits.argmax().item()
        top3 = torch.topk(logits, 3, dim=-1).indices[0].cpu().numpy()
    
    correct += (pred == target)
    top3_correct += (target in top3)
    total += 1

print(f"\n=== LSTM Diagnostic ===")
print(f"Top-1 Accuracy:  {correct/total:.4f}  (random = {1/config.NUM_BANDS:.4f})")
print(f"Top-3 Accuracy:  {top3_correct/total:.4f}  (random = {3/config.NUM_BANDS:.4f})")

if correct/total > 0.30:
    print("✅ LSTM learned the pattern!")
elif correct/total > 0.10:
    print("⚠️  LSTM learned weakly — try more training epochs")
else:
    print("❌ LSTM did not learn — likely seed/algorithm mismatch")
    print("    The Markov transition matrix during training might differ from now.")