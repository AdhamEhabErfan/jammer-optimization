import matplotlib.pyplot as plt
import numpy as np
import torch
from config import Config
from frequency_hopping import FrequencyHoppingTransmitter
from models.hybrid_model import HybridJammerNet


def visualize_jamming(config, num_steps=100):
    """Visualize jammer power allocation vs. true TX bands over time"""
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    
    # Load hybrid model
    model = HybridJammerNet(config.NUM_BANDS, config.SENSING_WINDOW,
                             config.LSTM_HIDDEN_SIZE).to(device)
    model.load_state_dict(torch.load('hybrid_jammer.pth'))
    model.eval()
    
    # Generate sequence
    tx = FrequencyHoppingTransmitter(config.NUM_BANDS, config.FH_ALGORITHM, config.FH_SEED)
    sequence = tx.generate_sequence(num_steps + config.SENSING_WINDOW)
    
    # Get power allocations
    power_matrix = np.zeros((num_steps, config.NUM_BANDS))
    true_bands = []
    
    for i in range(num_steps):
        window = sequence[i:i + config.SENSING_WINDOW]
        window_onehot = np.eye(config.NUM_BANDS)[window].astype(np.float32)
        x = torch.from_numpy(window_onehot).unsqueeze(0).to(device)
        
        with torch.no_grad():
            _, _, power = model(x)
        
        power_matrix[i] = power.cpu().numpy()[0]
        true_bands.append(sequence[i + config.SENSING_WINDOW])
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Heatmap of jammer power
    im = axes[0].imshow(power_matrix.T, aspect='auto', cmap='hot', origin='lower')
    axes[0].plot(range(num_steps), true_bands, 'co', markersize=4, 
                 label='True TX Band', alpha=0.7)
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Frequency Band')
    axes[0].set_title('Jammer Power Allocation Heatmap (TX bands shown as cyan dots)')
    axes[0].legend()
    plt.colorbar(im, ax=axes[0], label='Power')
    
    # Power on true band over time
    power_on_true = [power_matrix[t, true_bands[t]] for t in range(num_steps)]
    axes[1].plot(power_on_true, 'b-', linewidth=1.5)
    axes[1].axhline(y=1.0/config.NUM_BANDS, color='r', linestyle='--', 
                    label=f'Random baseline (1/{config.NUM_BANDS})')
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Power on True TX Band')
    axes[1].set_title('Jamming Effectiveness Over Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('jamming_visualization.png', dpi=150)
    plt.show()
    
    print(f"Avg power on true band: {np.mean(power_on_true):.4f}")
    print(f"Random baseline:        {1.0/config.NUM_BANDS:.4f}")
    print(f"Improvement factor:     {np.mean(power_on_true) * config.NUM_BANDS:.2f}x")


if __name__ == "__main__":
    config = Config()
    visualize_jamming(config)