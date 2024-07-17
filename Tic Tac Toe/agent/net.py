""" Nueral Network Model for the agent """

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.conv2(x)
        x = self.bn2(x)
        x += residual
        x = F.relu(x)
        return x

class QNetwork(nn.Module):
    def __init__(self, in_channels, action_space, num_blocks=3, hidden_channels=64, policy_head_channels=32, value_head_channels=3):
        super(QNetwork, self).__init__()
        self.first_block = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU()
        )

        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_channels, hidden_channels) for _ in range(num_blocks)
        ])

        self.policy_head = nn.Sequential(
            nn.Conv2d(hidden_channels, policy_head_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(policy_head_channels),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(policy_head_channels * action_space, action_space),
            nn.Softmax(dim=1)
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(hidden_channels, value_head_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(value_head_channels),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(value_head_channels * action_space, 1),
            nn.Tanh()
        )

    def forward(self, state):
        x = self.first_block(state)
        for block in self.blocks:
            x = block(x)

        policy = self.policy_head(x)
        value = self.value_head(x)

        return policy, value
