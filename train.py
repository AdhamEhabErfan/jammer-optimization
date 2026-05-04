import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from config import Config
from frequency_hopping import FrequencyHoppingTransmitter
from jammer_environment import JammerEnvironment
from models.lstm_predictor import LSTMPredictor
from models.dqn_agent import DQNAgent
from models.hybrid_model import HybridJammerNet


def generate_training_data(config, num_sequences=5000):
    """Generate supervised training data for LSTM predictor"""
    tx = FrequencyHoppingTransmitter(
        num_bands=config.NUM_BANDS,
        algorithm=config.FH_ALGORITHM,
        seed=config.FH_SEED
    )
    
    # Generate long sequence
    sequence = tx.generate_sequence(num_sequences + config.SENSING_WINDOW)
    
    X, y = [], []
    for i in range(len(sequence) - config.SENSING_WINDOW):
        # Input: window of one-hot encoded bands
        window = sequence[i:i + config.SENSING_WINDOW]
        window_onehot = np.eye(config.NUM_BANDS)[window]
        X.append(window_onehot)
        # Target: next band
        y.append(sequence[i + config.SENSING_WINDOW])
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def train_lstm_predictor(config):
    """Phase 1: Train LSTM to predict frequency hopping pattern"""
    print("=" * 60)
    print("PHASE 1: Training LSTM Frequency Predictor")
    print("=" * 60)
    
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    
    # Generate data
    X, y = generate_training_data(config, num_sequences=10000)
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=64, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=64
    )
    
    model = LSTMPredictor(
        num_bands=config.NUM_BANDS,
        hidden_size=config.LSTM_HIDDEN_SIZE,
        num_layers=config.LSTM_NUM_LAYERS,
        dropout=config.DROPOUT
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    train_losses, val_accs = [], []
    
    for epoch in range(50):
        model.train()
        epoch_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).argmax(dim=-1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        
        val_acc = correct / total
        train_losses.append(epoch_loss / len(train_loader))
        val_accs.append(val_acc)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/50 | Loss: {train_losses[-1]:.4f} | Val Acc: {val_acc:.4f}")
    
    torch.save(model.state_dict(), 'lstm_predictor.pth')
    print(f"\n✓ LSTM saved. Final accuracy: {val_accs[-1]:.4f}")
    print(f"  (Random baseline: {1.0/config.NUM_BANDS:.4f})")
    
    return model, train_losses, val_accs


def train_dqn_jammer(config):
    """Phase 2: Train DQN to optimize jamming strategy"""
    print("\n" + "=" * 60)
    print("PHASE 2: Training DQN Jammer Agent")
    print("=" * 60)
    
    env = JammerEnvironment(config)
    agent = DQNAgent(config)
    
    episode_rewards = []
    jam_success_rates = []
    
    for episode in range(config.NUM_EPISODES):
        state = env.reset()
        total_reward = 0
        jams = 0
        steps = 0
        
        for step in range(config.EPISODE_LENGTH):
            action = agent.select_action(state, training=True)
            next_state, reward, done, info = env.step(action)
            
            agent.replay_buffer.push(state, action, reward, next_state, done)
            agent.train_step()
            
            state = next_state
            total_reward += reward
            jams += int(info['jammed'])
            steps += 1
            
            if done:
                break
        
        success_rate = jams / steps
        episode_rewards.append(total_reward)
        jam_success_rates.append(success_rate)
        
        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:])
            avg_success = np.mean(jam_success_rates[-20:])
            print(f"Episode {episode+1}/{config.NUM_EPISODES} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Jam Success: {avg_success:.3f} | "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    torch.save(agent.q_network.state_dict(), 'dqn_jammer.pth')
    print("\n✓ DQN agent saved.")
    
    return agent, episode_rewards, jam_success_rates


def train_hybrid(config):
    """Phase 3: Train hybrid model with combined loss"""
    print("\n" + "=" * 60)
    print("PHASE 3: Training Hybrid Jammer Network")
    print("=" * 60)
    
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    X, y = generate_training_data(config, num_sequences=10000)
    
    split = int(0.8 * len(X))
    X_train = torch.from_numpy(X[:split]).to(device)
    y_train = torch.from_numpy(y[:split]).to(device)
    X_val = torch.from_numpy(X[split:]).to(device)
    y_val = torch.from_numpy(y[split:]).to(device)
    
    model = HybridJammerNet(
        num_bands=config.NUM_BANDS,
        sensing_window=config.SENSING_WINDOW,
        hidden_size=config.LSTM_HIDDEN_SIZE
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    for epoch in range(50):
        model.train()
        # Mini-batches
        perm = torch.randperm(len(X_train))
        losses = []
        for i in range(0, len(X_train), 64):
            idx = perm[i:i+64]
            xb, yb = X_train[idx], y_train[idx]
            
            pred_logits, pred_probs, power_alloc = model(xb)
            
            # Loss 1: Prediction accuracy
            pred_loss = nn.CrossEntropyLoss()(pred_logits, yb)
            
            # Loss 2: Power should match true band (jamming reward)
            target_onehot = F.one_hot(yb, config.NUM_BANDS).float()
            # Maximize power on true band
            power_reward = -(power_alloc * target_onehot).sum(dim=-1).mean()
            
            # Entropy regularization (avoid concentrating all power on one band)
            entropy_reg = -(power_alloc * torch.log(power_alloc + 1e-10)).sum(dim=-1).mean()
            
            loss = pred_loss + power_reward - 0.01 * entropy_reg
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        # Validation
        model.eval()
        with torch.no_grad():
            _, val_probs, val_power = model(X_val)
            val_acc = (val_probs.argmax(dim=-1) == y_val).float().mean().item()
            # Power on true band
            target_onehot = F.one_hot(y_val, config.NUM_BANDS).float()
            avg_power_on_true = (val_power * target_onehot).sum(dim=-1).mean().item()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/50 | Loss: {np.mean(losses):.4f} | "
                  f"Pred Acc: {val_acc:.4f} | "
                  f"Avg Power on True Band: {avg_power_on_true:.4f}")
    
    torch.save(model.state_dict(), 'hybrid_jammer.pth')
    print("\n✓ Hybrid model saved.")
    return model


import torch.nn.functional as F

if __name__ == "__main__":
    config = Config()
    
    # Phase 1: Train LSTM predictor
    lstm_model, losses, accs = train_lstm_predictor(config)
    
    # Phase 2: Train DQN agent
    dqn_agent, rewards, success = train_dqn_jammer(config)
    
    # Phase 3: Train hybrid model
    hybrid_model = train_hybrid(config)
    
    print("\n" + "=" * 60)
    print("✓ All training phases complete!")
    print("=" * 60)