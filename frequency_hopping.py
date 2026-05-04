import numpy as np
from scipy.stats import entropy

class FrequencyHoppingTransmitter:
    """Simulates frequency hopping communication system"""
    
    def __init__(self, num_bands, algorithm='pseudo_random', seed=42):
        self.num_bands = num_bands
        self.algorithm = algorithm
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.current_band = 0
        self.time_step = 0
        
        # For Markov-based hopping
        if algorithm == 'markov':
            self.transition_matrix = self._build_markov_matrix()
        
        # For chaotic hopping (logistic map)
        if algorithm == 'chaotic':
            self.chaos_x = 0.1
            self.chaos_r = 3.99
    
    def _build_markov_matrix(self):
        """Build a transition probability matrix for Markov-based hopping"""
        matrix = self.rng.dirichlet(np.ones(self.num_bands), size=self.num_bands)
        return matrix
    
    def get_next_band(self):
        """Generate next frequency band based on algorithm"""
        if self.algorithm == 'pseudo_random':
            band = self.rng.randint(0, self.num_bands)
        
        elif self.algorithm == 'chaotic':
            # Logistic map for chaotic sequence
            self.chaos_x = self.chaos_r * self.chaos_x * (1 - self.chaos_x)
            band = int(self.chaos_x * self.num_bands) % self.num_bands
        
        elif self.algorithm == 'markov':
            probs = self.transition_matrix[self.current_band]
            band = self.rng.choice(self.num_bands, p=probs)
        
        elif self.algorithm == 'adaptive':
            # Adaptive: avoid recently used bands
            band = self.rng.randint(0, self.num_bands)
            # Add adaptive logic based on jamming feedback (advanced)
        
        else:
            band = self.rng.randint(0, self.num_bands)
        
        self.current_band = band
        self.time_step += 1
        return band
    
    def generate_sequence(self, length):
        """Generate a full hopping sequence"""
        return np.array([self.get_next_band() for _ in range(length)])
    
    def reset(self):
        self.rng = np.random.RandomState(self.seed)
        self.current_band = 0
        self.time_step = 0
        if self.algorithm == 'chaotic':
            self.chaos_x = 0.1


class Channel:
    """Communication channel with noise and jamming effects"""
    
    def __init__(self, num_bands, snr_db=10, noise_power=0.01):
        self.num_bands = num_bands
        self.snr_db = snr_db
        self.noise_power = noise_power
        self.signal_power = noise_power * (10 ** (snr_db / 10))
    
    def compute_sinr(self, tx_band, jammer_power_allocation):
        """Compute SINR for the transmitter's current band"""
        jammer_power_on_band = jammer_power_allocation[tx_band]
        sinr = self.signal_power / (self.noise_power + jammer_power_on_band)
        return sinr
    
    def is_jammed(self, tx_band, jammer_power_allocation, threshold=1.0):
        """Determine if communication is successfully jammed"""
        sinr = self.compute_sinr(tx_band, jammer_power_allocation)
        sinr_db = 10 * np.log10(sinr + 1e-10)
        return sinr_db < threshold  # Jammed if SINR drops below threshold