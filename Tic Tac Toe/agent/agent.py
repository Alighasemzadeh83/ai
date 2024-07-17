""" Agent class for the game. """

import torch

class Agent:
    def __init__(self, model, transform):
        self.model = model
        self.transform = transform

    def get_policy_value(self, state):
        self.model.eval()
        with torch.no_grad():
            policy, value = self.model(self.transform(state).unsqueeze(0))
        policy = policy.squeeze().cpu().numpy()
        value = value.item()
        return policy, value

    def act(self, state):
        policy, _ = self.get_policy_value(state)
        return policy.argmax()
        