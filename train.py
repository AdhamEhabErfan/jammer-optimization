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
    print(f"🖥️  Using device: {device}")
    
    # ⭐ MORE training data
    X, y = generate_training_data(config, num_sequences=50000)  # ← was 10000
    
    split = int(0.85 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    print(f"📊 Training samples: {len(X_train)}, Validation: {len(X_val)}")
    
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=128, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=256
    )
    
    model = LSTMPredictor(
        num_bands=config.NUM_BANDS,
        hidden_size=config.LSTM_HIDDEN_SIZE,
        num_layers=config.LSTM_NUM_LAYERS,
        dropout=config.DROPOUT
    ).to(device)
    
    # ⭐ Lower learning rate + scheduler
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    criterion = nn.CrossEntropyLoss()
    
    train_losses, val_accs = [], []
    best_acc = 0
    
    NUM_EPOCHS = 100  # ⭐ More epochs
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            # ⭐ Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        scheduler.step()
        
        # Validation
        model.eval()
        correct, total = 0, 0
        top3_correct = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                preds = logits.argmax(dim=-1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
                # Top-3 accuracy
                top3 = torch.topk(logits, 3, dim=-1).indices
                top3_correct += (top3 == yb.unsqueeze(1)).any(dim=1).sum().item()
        
        val_acc = correct / total
        top3_acc = top3_correct / total
        train_losses.append(epoch_loss / len(train_loader))
        val_accs.append(val_acc)
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'lstm_predictor.pth')
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {train_losses[-1]:.4f} | "
                  f"Top-1: {val_acc:.4f} | Top-3: {top3_acc:.4f} | "
                  f"Best: {best_acc:.4f}")
    
    print(f"\n✓ LSTM saved (best top-1 acc: {best_acc:.4f})")
    print(f"  Theoretical max: ~0.54  |  Random: {1.0/config.NUM_BANDS:.4f}")
    
    return model, train_losses, val_accs


def train_dqn_jammer(config):
    """Phase 2: Train DQN with improved reward and longer training"""
    print("\n" + "=" * 60)
    print("PHASE 2: Training DQN Jammer Agent")
    print("=" * 60)
    
    env = JammerEnvironment(config)
    agent = DQNAgent(config)
    
    episode_rewards = []
    jam_success_rates = []
    
    NUM_EPISODES = 1500  # ⭐ More episodes
    
    for episode in range(NUM_EPISODES):
        # ⭐ Different seed per episode for diversity
        state = env.reset(episode_seed=config.FH_SEED + episode * 13)
        total_reward = 0
        jams = 0
        steps = 0
        hits = 0  # Direct hits
        
        for step in range(config.EPISODE_LENGTH):
            action = agent.select_action(state, training=True)
            next_state, reward, done, info = env.step(action)
            
            agent.replay_buffer.push(state, action, reward, next_state, done)
            agent.train_step()
            
            state = next_state
            total_reward += reward
            jams += int(info['jammed'])
            hits += int(action == info['true_band'])
            steps += 1
            
            if done:
                break
        
        success_rate = jams / steps
        hit_rate = hits / steps
        episode_rewards.append(total_reward)
        jam_success_rates.append(success_rate)
        
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            avg_success = np.mean(jam_success_rates[-50:])
            print(f"Episode {episode+1}/{NUM_EPISODES} | "
                  f"Reward: {avg_reward:.2f} | "
                  f"Jam Rate: {avg_success:.3f} | "
                  f"Hit Rate: {hit_rate:.3f} | "
                  f"ε: {agent.epsilon:.3f}")
    
    torch.save(agent.q_network.state_dict(), 'dqn_jammer.pth')
    print("\n✓ DQN agent saved.")
    
    return agent, episode_rewards, jam_success_rates

def train_hybrid(config):
    """Phase 3: Train hybrid model with combined loss"""
    print("\n" + "=" * 60)
    print("PHASE 3: Training Hybrid Jammer Network")
    print("=" * 60)
    
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    
    # ⭐ MORE data
    X, y = generate_training_data(config, num_sequences=50000)
    
    split = int(0.85 * len(X))
    X_train = torch.from_numpy(X[:split]).to(device)
    y_train = torch.from_numpy(y[:split]).to(device)
    X_val = torch.from_numpy(X[split:]).to(device)
    y_val = torch.from_numpy(y[split:]).to(device)
    
    model = HybridJammerNet(
        num_bands=config.NUM_BANDS,
        sensing_window=config.SENSING_WINDOW,
        hidden_size=config.LSTM_HIDDEN_SIZE
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    NUM_EPOCHS = 100
    best_jam_score = 0
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        perm = torch.randperm(len(X_train))
        losses = []
        for i in range(0, len(X_train), 128):
            idx = perm[i:i+128]
            xb, yb = X_train[idx], y_train[idx]
            
            pred_logits, pred_probs, power_alloc = model(xb)
            
            # Loss 1: Prediction accuracy
            pred_loss = nn.CrossEntropyLoss()(pred_logits, yb)
            
            # Loss 2: Maximize power on true band
            target_onehot = F.one_hot(yb, config.NUM_BANDS).float()
            power_reward = -(power_alloc * target_onehot).sum(dim=-1).mean()
            
            # Loss 3: Light entropy regularization (avoid total collapse)
            entropy_reg = -(power_alloc * torch.log(power_alloc + 1e-10)).sum(dim=-1).mean()
            
            # ⭐ Better balance — emphasize prediction
            loss = 2.0 * pred_loss + power_reward - 0.001 * entropy_reg
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        
        scheduler.step()
        
        model.eval()
        with torch.no_grad():
            _, val_probs, val_power = model(X_val)
            val_acc = (val_probs.argmax(dim=-1) == y_val).float().mean().item()
            target_onehot = F.one_hot(y_val, config.NUM_BANDS).float()
            avg_power_on_true = (val_power * target_onehot).sum(dim=-1).mean().item()
        
        # Save best
        if avg_power_on_true > best_jam_score:
            best_jam_score = avg_power_on_true
            torch.save(model.state_dict(), 'hybrid_jammer.pth')
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {np.mean(losses):.4f} | "
                  f"Pred Acc: {val_acc:.4f} | "
                  f"Power on True: {avg_power_on_true:.4f} | "
                  f"Best: {best_jam_score:.4f}")
    
    print(f"\n✓ Hybrid saved (best power score: {best_jam_score:.4f})")
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