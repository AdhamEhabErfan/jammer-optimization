import numpy as np
from frequency_hopping import FrequencyHoppingTransmitter, Channel


class JammerEnvironment:
    """RL Environment for the jammer agent (no gym dependency)"""
    
    def __init__(self, config):
        self.config = config
        self.num_bands = config.NUM_BANDS
        self.sensing_window = config.SENSING_WINDOW
        self.num_jammer_bands = config.NUM_JAMMER_BANDS
        
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
        self.episode_count = 0
    
    def reset(self, episode_seed=None):
        """Reset with a different starting state each episode"""
        self.episode_count += 1
        if episode_seed is None:
            # Use episode count to vary the starting state
            episode_seed = self.config.FH_SEED + self.episode_count * 1000
        
        self.tx.reset(new_seed=episode_seed)
        self.history = np.zeros((self.sensing_window, self.num_bands), dtype=np.float32)
        self.current_step = 0
        
        for i in range(self.sensing_window):
            band = self.tx.get_next_band()
            self.history[i, band] = 1.0
        
        return self.history.copy()
    
    def step(self, action):
        true_band = self.tx.get_next_band()
        
        # Build power allocation
        if isinstance(action, (int, np.integer)):
            power_allocation = np.zeros(self.num_bands)
            power_allocation[action] = self.config.JAMMER_TOTAL_POWER
        else:
            action = np.asarray(action, dtype=np.float32).flatten()
            # Normalize to sum to JAMMER_TOTAL_POWER
            action = action / (action.sum() + 1e-10)
            power_allocation = action * self.config.JAMMER_TOTAL_POWER
        
        sinr = self.channel.compute_sinr(true_band, power_allocation)
        sinr_db = 10 * np.log10(sinr + 1e-10)
        
        # Improved reward: more sensitive to power on true band
        power_on_true = power_allocation[true_band]
        reward = power_on_true * 10.0 - sinr_db / 10.0
        
        if isinstance(action, (int, np.integer)) and action == true_band:
            reward += 5.0
        
        self.history = np.roll(self.history, -1, axis=0)
        self.history[-1] = 0
        self.history[-1, true_band] = 1.0
        
        self.current_step += 1
        done = self.current_step >= self.config.EPISODE_LENGTH
        
        info = {
            'true_band': int(true_band),
            'sinr_db': float(sinr_db),
            'jammed': bool(self.channel.is_jammed(true_band, power_allocation)),
            'power_on_true': float(power_on_true)
        }
        
        return self.history.copy(), reward, done, info