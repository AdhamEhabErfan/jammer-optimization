import torch
import numpy as np
import matplotlib.pyplot as plt
from config import Config
from jammer_environment import JammerEnvironment
from models.lstm_predictor import LSTMPredictor
from models.hybrid_model import HybridJammerNet


def evaluate_strategies(config, num_episodes=50):
    """Compare jamming strategies"""
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    env = JammerEnvironment(config)
    
    # Load models
    lstm_model = LSTMPredictor(config.NUM_BANDS, config.LSTM_HIDDEN_SIZE,
                                config.LSTM_NUM_LAYERS, config.DROPOUT).to(device)
    lstm_model.load_state_dict(torch.load('lstm_predictor.pth'))
    lstm_model.eval()
    
    hybrid_model = HybridJammerNet(config.NUM_BANDS, config.SENSING_WINDOW,
                                    config.LSTM_HIDDEN_SIZE).to(device)
    hybrid_model.load_state_dict(torch.load('hybrid_jammer.pth'))
    hybrid_model.eval()
    
    strategies = {
        'Random': lambda s: np.random.randint(0, config.NUM_BANDS),
        'Uniform_Spread': lambda s: np.ones(config.NUM_BANDS) / config.NUM_BANDS,
        'LSTM_Predictor': None,   # uses model
        'Hybrid_NN': None,        # uses model
    }
    
    results = {name: [] for name in strategies}
    
    for strategy_name in strategies:
        for ep in range(num_episodes):
            state = env.reset()
            jams = 0
            for step in range(config.EPISODE_LENGTH):
                if strategy_name == 'Random':
                    action = np.random.randint(0, config.NUM_BANDS)
                elif strategy_name == 'Uniform_Spread':
                    action = np.ones(config.NUM_BANDS) / config.NUM_BANDS
                elif strategy_name == 'LSTM_Predictor':
                    with torch.no_grad():
                        s_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                        action = lstm_model(s_t).argmax(dim=-1).item()
                elif strategy_name == 'Hybrid_NN':
                    with torch.no_grad():
                        s_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                        _, _, power = hybrid_model(s_t)
                        action = power.cpu().numpy()[0]
                
                state, reward, done, info = env.step(action)
                jams += int(info['jammed'])
                if done:
                    break
            
            results[strategy_name].append(jams / config.EPISODE_LENGTH)
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS (Jamming Success Rate)")
    print("=" * 60)
    for name, scores in results.items():
        print(f"{name:20s}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot([results[k] for k in strategies], labels=list(strategies.keys()))
    ax.set_ylabel('Jamming Success Rate')
    ax.set_title('Comparison of Jamming Strategies')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('strategy_comparison.png', dpi=150)
    plt.show()
    
    return results


if __name__ == "__main__":
    config = Config()
    results = evaluate_strategies(config)