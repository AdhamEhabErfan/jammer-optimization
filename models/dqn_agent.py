import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class DQN(nn.Module):
    """Deep Q-Network for power allocation strategy"""
    
    def __init__(self, num_bands, sensing_window, hidden_size=128):
        super().__init__()
        input_dim = num_bands * sensing_window
        
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_bands)  # Q-values for each band
        )
    
    def forward(self, x):
        return self.network(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, config):
        self.config = config
        self.num_bands = config.NUM_BANDS
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
        
        self.q_network = DQN(
            config.NUM_BANDS, config.SENSING_WINDOW
        ).to(self.device)
        self.target_network = DQN(
            config.NUM_BANDS, config.SENSING_WINDOW
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.LEARNING_RATE)
        self.replay_buffer = ReplayBuffer(config.REPLAY_BUFFER_SIZE)
        
        self.epsilon = config.EPSILON_START
        self.steps = 0
    
    def select_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randint(0, self.num_bands - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def train_step(self):
        if len(self.replay_buffer) < self.config.BATCH_SIZE:
            return None
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.config.BATCH_SIZE
        )
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Target Q values
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + self.config.GAMMA * next_q * (1 - dones)
        
        # Loss
        loss = nn.MSELoss()(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        self.steps += 1
        if self.steps % self.config.TARGET_UPDATE_FREQ == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        self.epsilon = max(
            self.config.EPSILON_END,
            self.epsilon * self.config.EPSILON_DECAY
        )
        
        return loss.item()