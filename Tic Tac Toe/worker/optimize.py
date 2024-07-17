"""
Optimize the model using the experiences generated from self-play
"""

import torch
from torch.utils.data import DataLoader, Dataset

import os
import numpy as np


class ExperienceDataset(Dataset):
    def __init__(self, path=None, transform=None):
        if path is None:
            path = find_newest_dataset()
        self.transform = transform
        data = np.load(path)
        self.states = data["states"]
        self.policies = data["policies"]
        self.values = data["values"]

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        state = torch.from_numpy(self.states[idx], dtype=torch.float32)
        if self.transform:
            state = self.transform(state)
        policy = torch.from_numpy(self.policies[idx], dtype=torch.float32)
        value = torch.tensor(self.values[idx], dtype=torch.float32).unsqueeze(0)
        return state, policy, value

def find_newest_dataset():
    newest = None
    for root, dirs, files in os.walk("experiences"):
        for file in files:
            if file.endswith(".npz"):
                path = os.path.join(root, file)
                timestamp = int(os.path.basename(root))
                if newest is None or timestamp > newest[0]:
                    newest = (timestamp, path)
    return newest[1]
