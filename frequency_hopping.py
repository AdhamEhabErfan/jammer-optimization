import numpy as np


class FrequencyHoppingTransmitter:
    def __init__(self, num_bands, algorithm='pseudo_random', seed=42):
        self.num_bands = num_bands
        self.algorithm = algorithm
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.current_band = 0
        self.time_step = 0
        
        # ⭐ MUCH more structured Markov chain
        if algorithm == 'markov':
            fixed_rng = np.random.RandomState(seed)
            # Use small alpha → very peaked distributions = strong patterns
            self.transition_matrix = fixed_rng.dirichlet(
                np.ones(self.num_bands) * 0.05,  # ← was 0.5, now 0.05 (10× more peaked)
                size=self.num_bands
            )
        
        # ⭐ NEW: Periodic pattern (very learnable - good for demo!)
        if algorithm == 'periodic':
            fixed_rng = np.random.RandomState(seed)
            # Generate a fixed sequence of length 50 that repeats
            self.period_length = 50
            self.periodic_seq = fixed_rng.randint(0, num_bands, size=self.period_length)
        
        # ⭐ NEW: LFSR-like pattern
        if algorithm == 'lfsr':
            self.lfsr_state = (seed % (2**8 - 1)) + 1  # 8-bit LFSR
        
        if algorithm == 'chaotic':
            self.chaos_x = 0.1 + (seed % 100) / 1000.0
            self.chaos_r = 3.99
    
    def get_next_band(self):
        if self.algorithm == 'pseudo_random':
            band = self.rng.randint(0, self.num_bands)
        
        elif self.algorithm == 'chaotic':
            self.chaos_x = self.chaos_r * self.chaos_x * (1 - self.chaos_x)
            band = int(self.chaos_x * self.num_bands) % self.num_bands
        
        elif self.algorithm == 'markov':
            probs = self.transition_matrix[self.current_band]
            band = self.rng.choice(self.num_bands, p=probs)
        
        elif self.algorithm == 'periodic':
            # Very learnable: deterministic repeating sequence
            band = int(self.periodic_seq[self.time_step % self.period_length])
        
        elif self.algorithm == 'lfsr':
            # Linear Feedback Shift Register (real FH systems use this!)
            bit = ((self.lfsr_state >> 0) ^ (self.lfsr_state >> 2) ^ 
                   (self.lfsr_state >> 3) ^ (self.lfsr_state >> 4)) & 1
            self.lfsr_state = (self.lfsr_state >> 1) | (bit << 7)
            band = self.lfsr_state % self.num_bands
        
        else:
            band = self.rng.randint(0, self.num_bands)
        
        self.current_band = band
        self.time_step += 1
        return band
    
    def generate_sequence(self, length):
        return np.array([self.get_next_band() for _ in range(length)])
    
    def reset(self, new_seed=None):
        if new_seed is not None:
            self.rng = np.random.RandomState(new_seed)
            if self.algorithm == 'chaotic':
                self.chaos_x = 0.1 + (new_seed % 100) / 1000.0
            if self.algorithm == 'lfsr':
                self.lfsr_state = (new_seed % (2**8 - 1)) + 1
        else:
            self.rng = np.random.RandomState(self.seed)
            if self.algorithm == 'chaotic':
                self.chaos_x = 0.1
            if self.algorithm == 'lfsr':
                self.lfsr_state = (self.seed % (2**8 - 1)) + 1
        self.current_band = self.rng.randint(0, self.num_bands)
        self.time_step = 0


class Channel:
    def __init__(self, num_bands, snr_db=10, noise_power=0.01):
        self.num_bands = num_bands
        self.snr_db = snr_db
        self.noise_power = noise_power
        self.signal_power = noise_power * (10 ** (snr_db / 10))
    
    def compute_sinr(self, tx_band, jammer_power_allocation):
        jammer_power_on_band = jammer_power_allocation[tx_band]
        sinr = self.signal_power / (self.noise_power + jammer_power_on_band)
        return sinr
    
    def is_jammed(self, tx_band, jammer_power_allocation, min_jam_power=0.15):
        """Jammed only if significant power on TX band AND SINR low"""
        power_on_true = jammer_power_allocation[tx_band]
        sinr = self.compute_sinr(tx_band, jammer_power_allocation)
        sinr_db = 10 * np.log10(sinr + 1e-10)
        return (power_on_true >= min_jam_power) and (sinr_db < 0.0)