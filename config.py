import torch

class Config:
    # === Auto-detect device ===
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Frequency hopping
    NUM_BANDS = 16
    HOP_DURATION = 1
    TOTAL_TIME_SLOTS = 10000
    
    FH_SEED = 42
    FH_ALGORITHM = 'markov'
    
    # Jammer
    JAMMER_TOTAL_POWER = 0.5
    NUM_JAMMER_BANDS = 3
    SENSING_WINDOW = 20
    
    # Neural Network — MUST MATCH COLAB!
    LSTM_HIDDEN_SIZE = 256       # ← changed from 128 to 256
    LSTM_NUM_LAYERS = 2
    DROPOUT = 0.2
    
    # DQN
    GAMMA = 0.95
    EPSILON_START = 1.0
    EPSILON_END = 0.05
    EPSILON_DECAY = 0.995
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 128
    REPLAY_BUFFER_SIZE = 20000
    TARGET_UPDATE_FREQ = 100
    
    # Training
    NUM_EPISODES = 800
    EPISODE_LENGTH = 200
    
    # Channel
    SNR_DB = 5
    NOISE_POWER = 0.01