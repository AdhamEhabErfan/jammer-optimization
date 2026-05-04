import numpy as np
import gym
from gym import spaces
from frequency_hopping import FrequencyHoppingTransmitter, Channel

class JammerEnvironment(gym.Env):
    """RL Environment for the jammer agent"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_bands = config.NUM_BANDS
        self.sensing_window = config.SENSING_WINDOW
        self.num_jammer_bands = config.NUM_JAMMER_BANDS
        
        # Action space: select which bands to jam (discrete combinations)
        # Simplified: pick top-K bands to allocate power
        self.action_space = spaces.Discrete(self.num_bands)
        
        # Observation: history of observed transmitter bands (one-hot)
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(self.sensing_window, self.num_bands),
            dtype=np.float32
        )
        
        self.tx = FrequencyHoppingTransmitter(
            num_bands=self.num_bands,
            algorithm=config.FH_ALGORITHM,
            seed=config.FH_SEED
        )
        self.channel = Channel(
            num_bands=self.num_bands,
            snr_db=config.SNR_DB,
            noise_power=config.NOISE_POWER
        )
        
        self.history = None
        self.current_step = 0
    
    def reset(self):
        self.tx.reset()
        self.history = np.zeros((self.sensing_window, self.num_bands), dtype=np.float32)
        self.current_step = 0
        
        # Fill initial history
        for i in range(self.sensing_window):
            band = self.tx.get_next_band()
            self.history[i, band] = 1.0
        
        return self.history.copy()
    
    def step(self, action):
        """
        action: predicted band(s) to jam
        Can be a single integer (top-1) or array of probabilities
        """
        # Get true next band from transmitter
        true_band = self.tx.get_next_band()
        
        # Build power allocation
        if isinstance(action, (int, np.integer)):
            # Single band targeting
            power_allocation = np.zeros(self.num_bands)
            power_allocation[action] = self.config.JAMMER_TOTAL_POWER
        else:
            # Distributed power allocation (probability vector)
            power_allocation = np.array(action) * self.config.JAMMER_TOTAL_POWER
        
        # Compute reward based on jamming success
        sinr = self.channel.compute_sinr(true_band, power_allocation)
        sinr_db = 10 * np.log10(sinr + 1e-10)
        
        # Reward: higher when SINR is lower (better jamming)
        reward = -sinr_db / 10.0  # Normalize
        
        # Bonus for direct hit
        if isinstance(action, (int, np.integer)) and action == true_band:
            reward += 5.0
        
        # Update history (sliding window)
        self.history = np.roll(self.history, -1, axis=0)
        self.history[-1] = 0
        self.history[-1, true_band] = 1.0
        
        self.current_step += 1
        done = self.current_step >= self.config.EPISODE_LENGTH
        
        info = {
            'true_band': true_band,
            'sinr_db': sinr_db,
            'jammed': sinr_db < 1.0
        }
        
        return self.history.copy(), reward, done, info