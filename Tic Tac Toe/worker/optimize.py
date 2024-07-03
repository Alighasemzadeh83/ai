"""
Optimize the model using the experiences generated from self-play
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from datetime import datetime

from agent.model import QNetwork

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Optimize:
    def __init__(self, model: QNetwork, experiences, gamma=0.99):
        self.model = model
        self.experiences = experiences
        self.gamma = gamma
    
    def optimize_model(self, num_epochs, batch_size):
        optimizer = optim.Adam(self.model.parameters())
        
        for epoch in range(num_epochs):
            states, actions, rewards, next_states, dones = self.sample_batch(batch_size)
            states = torch.from_numpy(states).float().to(device)
            actions = torch.from_numpy(actions).long().to(device)
            rewards = torch.from_numpy(rewards).float().to(device)
            next_states = torch.from_numpy(next_states).float().to(device)
            dones = torch.from_numpy(dones.astype(np.uint8)).float().to(device)
            
            self.model.train()
            Q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            next_Q_values = self.model(next_states).detach().max(1)[0]
            target_Q_values = rewards + (1 - dones) * self.gamma * next_Q_values
            
            loss = F.mse_loss(Q_values, target_Q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        directory = f"models/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        torch.save(self.model.state_dict(), directory)
    
    def sample_batch(self, batch_size):
        indices = np.random.choice(len(self.experiences), batch_size, replace=False)
        batch = [self.experiences[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
