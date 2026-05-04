import numpy as np
from frequency_hopping import FrequencyHoppingTransmitter
from config import Config

config = Config()

print(f"Analyzing predictability of: {config.FH_ALGORITHM}\n")

tx = FrequencyHoppingTransmitter(config.NUM_BANDS, config.FH_ALGORITHM, config.FH_SEED)
seq = tx.generate_sequence(50000)

# Bigram statistics: P(next | current)
bigrams = np.zeros((config.NUM_BANDS, config.NUM_BANDS))
for i in range(len(seq) - 1):
    bigrams[seq[i], seq[i+1]] += 1

# Normalize
row_sums = bigrams.sum(axis=1, keepdims=True)
bigrams = bigrams / (row_sums + 1e-10)

# Best you can do: pick argmax of P(next | current)
correct = 0
for i in range(len(seq) - 1):
    pred = bigrams[seq[i]].argmax()
    correct += (pred == seq[i+1])

theoretical_max = correct / (len(seq) - 1)

print(f"Random baseline:        {1/config.NUM_BANDS:.4f}")
print(f"Bigram (1-step) max:    {theoretical_max:.4f}")
print(f"\nIf LSTM gets close to {theoretical_max:.2f}, it's doing great!")

# Top-3 theoretical max
top3 = 0
for i in range(len(seq) - 1):
    top3_pred = np.argsort(bigrams[seq[i]])[-3:]
    top3 += (seq[i+1] in top3_pred)
print(f"Top-3 bigram max:       {top3/(len(seq)-1):.4f}")