""" Agent class for the game. """

import torch
import torch.nn.functional as F
import numpy as np
import random

from agent.model import QNetwork

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TAU = 1e-3

class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = QNetwork(state_size, action_size).to(device)
    
    def act(self, state, eps=0.0):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.model.eval()
        with torch.no_grad():
            action_values = self.model(state)
        self.model.train()
        
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
