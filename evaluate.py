import torch
import numpy as np
import matplotlib.pyplot as plt
from config import Config
from jammer_environment import JammerEnvironment
from models.lstm_predictor import LSTMPredictor
from models.hybrid_model import HybridJammerNet
from models.dqn_agent import DQN


def evaluate_strategies(config, num_episodes=50):
    # Auto-detect device — works on both GPU and CPU machines
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}\n")
    
    env = JammerEnvironment(config)
    
    # === Load LSTM ===
    lstm_model = LSTMPredictor(
        config.NUM_BANDS, config.LSTM_HIDDEN_SIZE,
        config.LSTM_NUM_LAYERS, config.DROPOUT
    ).to(device)
    lstm_model.load_state_dict(
        torch.load('lstm_predictor.pth', map_location=device)
    )
    lstm_model.eval()
    print("✅ LSTM loaded")
    
    # === Load Hybrid ===
    hybrid_model = HybridJammerNet(
        config.NUM_BANDS, config.SENSING_WINDOW, config.LSTM_HIDDEN_SIZE
    ).to(device)
    hybrid_model.load_state_dict(
        torch.load('hybrid_jammer.pth', map_location=device)
    )
    hybrid_model.eval()
    print("✅ Hybrid loaded")
    
    # === Load DQN ===
    dqn_net = DQN(config.NUM_BANDS, config.SENSING_WINDOW).to(device)
    dqn_net.load_state_dict(
        torch.load('dqn_jammer.pth', map_location=device)
    )
    dqn_net.eval()
    print("✅ DQN loaded\n")
    
    strategy_names = ['Random', 'Uniform_Spread', 'Top3_Spread',
                      'LSTM_Predictor', 'DQN_Agent', 'Hybrid_NN']
    
    results = {name: {'jam_rate': [], 'avg_power_on_true': []} 
               for name in strategy_names}
    
    print("Running evaluation...")
    for ep in range(num_episodes):
        ep_seed = 12345 + ep * 7
        
        for strategy_name in strategy_names:
            state = env.reset(episode_seed=ep_seed)
            jams = 0
            power_on_true_list = []
            
            for step in range(config.EPISODE_LENGTH):
                if strategy_name == 'Random':
                    action = np.random.randint(0, config.NUM_BANDS)
                
                elif strategy_name == 'Uniform_Spread':
                    action = np.ones(config.NUM_BANDS) / config.NUM_BANDS
                
                elif strategy_name == 'Top3_Spread':
                    with torch.no_grad():
                        s_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                        probs = torch.softmax(lstm_model(s_t), dim=-1)
                        top_vals, top_idx = torch.topk(probs, 3, dim=-1)
                    action = np.zeros(config.NUM_BANDS)
                    for v, i in zip(top_vals[0].cpu().numpy(), top_idx[0].cpu().numpy()):
                        action[i] = v
                    action = action / action.sum()
                
                elif strategy_name == 'LSTM_Predictor':
                    with torch.no_grad():
                        s_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                        action = lstm_model(s_t).argmax(dim=-1).item()
                
                elif strategy_name == 'DQN_Agent':
                    with torch.no_grad():
                        s_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                        action = dqn_net(s_t).argmax(dim=-1).item()
                
                elif strategy_name == 'Hybrid_NN':
                    with torch.no_grad():
                        s_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                        _, _, power = hybrid_model(s_t)
                        action = power.cpu().numpy()[0]
                
                state, reward, done, info = env.step(action)
                jams += int(info['jammed'])
                power_on_true_list.append(info['power_on_true'])
                
                if done:
                    break
            
            results[strategy_name]['jam_rate'].append(jams / config.EPISODE_LENGTH)
            results[strategy_name]['avg_power_on_true'].append(
                np.mean(power_on_true_list)
            )
        
        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1}/{num_episodes} done")
    
    # === Print results ===
    print("\n" + "=" * 75)
    print(f"{'Strategy':<20}{'Jam Rate':<22}{'Avg Power on True Band':<25}")
    print("=" * 75)
    for name in strategy_names:
        jr = results[name]['jam_rate']
        po = results[name]['avg_power_on_true']
        print(f"{name:<20}{np.mean(jr):.4f} ± {np.std(jr):.4f}      "
              f"{np.mean(po):.4f} ± {np.std(po):.4f}")
    print("=" * 75)
    print(f"Random baseline reference: {1.0/config.NUM_BANDS:.4f}\n")
    
    # === Plotting ===
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    jam_data = [results[k]['jam_rate'] for k in strategy_names]
    axes[0].boxplot(jam_data, labels=strategy_names)
    axes[0].axhline(y=1.0/config.NUM_BANDS, color='r', linestyle='--', 
                    label='Random baseline')
    axes[0].set_ylabel('Jamming Success Rate')
    axes[0].set_title('Jamming Success Rate Comparison')
    axes[0].tick_params(axis='x', rotation=30)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    power_data = [results[k]['avg_power_on_true'] for k in strategy_names]
    axes[1].boxplot(power_data, labels=strategy_names)
    axes[1].axhline(y=1.0/config.NUM_BANDS, color='r', linestyle='--',
                    label='Random baseline')
    axes[1].set_ylabel('Avg Power on TX Band')
    axes[1].set_title('Power Concentration on True Band')
    axes[1].tick_params(axis='x', rotation=30)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('strategy_comparison.png', dpi=150)
    plt.show()
    
    return results


if __name__ == "__main__":
    config = Config()
    results = evaluate_strategies(config)