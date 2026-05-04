import numpy as np

class Config:
    # Frequency hopping parameters
    NUM_BANDS = 16              # Total available frequency bands
    HOP_DURATION = 1            # Time slot duration (ms)
    TOTAL_TIME_SLOTS = 10000    # Simulation time
    
    # FH algorithm parameters
    FH_SEED = 42
    FH_ALGORITHM = 'pseudo_random'  # 'pseudo_random', 'chaotic', 'markov', 'adaptive'
    
    # Jammer parameters
    JAMMER_TOTAL_POWER = 1.0    # Normalized total power
    NUM_JAMMER_BANDS = 3        # Bands jammer can simultaneously target
    SENSING_WINDOW = 20         # History window for NN input
    
    # Neural Network parameters
    LSTM_HIDDEN_SIZE = 128
    LSTM_NUM_LAYERS = 2
    DROPOUT = 0.2
    
    # DQN parameters
    GAMMA = 0.95
    EPSILON_START = 1.0
    EPSILON_END = 0.05
    EPSILON_DECAY = 0.995
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 64
    REPLAY_BUFFER_SIZE = 10000
    TARGET_UPDATE_FREQ = 100
    
    # Training
    NUM_EPISODES = 500
    EPISODE_LENGTH = 200
    DEVICE = 'cuda'  # or 'cpu'
    
    # Channel parameters
    SNR_DB = 10
    NOISE_POWER = 0.01