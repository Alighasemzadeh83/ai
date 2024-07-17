"""
Optimize the model using the experiences generated from self-play
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import os
import numpy as np
from tqdm import trange

def train(model, transform, lr, num_epochs, batch_size):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataloader = get_dataloader(batch_size, transform)
    for _ in trange(num_epochs):
        optimize(model, dataloader, optimizer)
    
    # save the model
    os.makedirs("models", exist_ok=True)
    # find the last generation
    generation = 0
    for root, dirs, files in os.walk("models"):
        for file in files:
            if file.endswith(".pth"):
                gen = int(file.split("_")[1].split(".")[0])
                if gen > generation:
                    generation = gen
    torch.save(model.state_dict(), f"models/generation_{generation + 1}.pth")


def load_best_model(model):
    os.makedirs("models", exist_ok=True)
    generation = 0
    for root, dirs, files in os.walk("models"):
        for file in files:
            if file.endswith(".pth"):
                gen = int(file.split("_")[1].split(".")[0])
                if gen > generation:
                    generation = gen
    if generation > 0:
        model.load_state_dict(torch.load(f"models/generation_{generation}.pth"))
    return model

def optimize(model, dataloader, optimizer):
    model.train()
    for state, policy, value in dataloader:
        optimizer.zero_grad()
        policy_pred, value_pred = model(state)
        policy_loss = F.cross_entropy(policy_pred, policy)
        value_loss = F.mse_loss(value_pred, value)
        loss = policy_loss + value_loss
        loss.backward()
        optimizer.step()

def get_dataloader(batch_size, transform):
    dataset = ExperienceDataset(transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

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
        state = torch.from_numpy(self.states[idx]).float()
        state = self.transform(state)
        policy = torch.from_numpy(self.policies[idx]).float()
        value = torch.tensor(self.values[idx], dtype=torch.float32).unsqueeze(0)
        return state, policy, value

def find_newest_dataset():
    newest = None
    for root, dirs, files in os.walk("experiences"):
        for file in files:
            if file.endswith(".npz"):
                if newest is None or file > newest:
                    newest = file
    return f"experiences/{newest}"
