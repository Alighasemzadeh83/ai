"""
Generate experiences based on self-play.

This class is used to generate experiences based on self-play.

Each experience is in format (state, action, value). The value is 1 if the agent wins, -1 if the agent loses, and 0 if it is a draw.
"""

import torch
import os
import datetime
import numpy as np
from tqdm import trange

def self_play(model, env, num_games, num_simulations, C, tau, epsilon):
    """
    Plays multiple games and save in file.
    """

    states = []
    policies = []
    values = []

    for _ in trange(num_games):
        state, policy, value = __self_play(model, env, num_simulations, C, tau, epsilon)
        states.append(state)
        policies.append(policy)
        values.append(value)

    states = np.concatenate(states)
    policies = np.concatenate(policies)
    values = np.concatenate(values)

    # save the experiences
    os.makedirs("experiences", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    np.savez(f"experiences/{timestamp}.npz", states=states, policies=policies, values=values)

def __self_play(model, env, num_simulations, C, tau, epsilon):
    """
    Plays one game and returns the experiences.
    """
    model.eval()
    memory = []
    mcts = MonteCarloTreeSearch(model, env, C, num_simulations, tau, epsilon)
    state = env.reset()
    next_state = None
    done = False
    reward = None

    while not done:
        with torch.no_grad():
            action_probs = mcts.search(state)
        action = np.random.choice(env.action_space, p=action_probs)
        next_state, reward, done = env.step(action)
        memory.append((state, action_probs, reward))
        state = next_state
    
    states = np.array([state for state, _, _ in memory], dtype=np.int8)
    policies = np.array([policy for _, policy, _ in memory], dtype=np.float32)
    values = np.full(len(memory), reward, dtype=np.int8)
    # set value for loser to -reward
    values[len(memory)%2::2] = -reward

    return states, policies, values


class Node:
    def __init__(self, state, value, done, parent=None, action_taken=None, prior=0):
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior
        self.value = value
        self.done = done

        self.children = []

        self.visit_count = 0
        self.value_sum = 0

class MonteCarloTreeSearch:
    def __init__(self, model, env, C, num_simulations, tau, epsilon):
        self.model = model
        self.env = env
        self.C = C
        self.num_simulations = num_simulations
        self.tau = tau
        self.epsilon = epsilon

    def search(self, state):
        """
        Returns the policy based on the MCTS
        """
        root = Node(state, 0, False)

        for _ in range(self.num_simulations):
            node = root

            while node.children:
                node = self._select(node)

            value = node.value

            if not node.done:
                policy, value = self.model(
                    self.env.transform(node.state).unsqueeze(0)
                )
                policy = policy.squeeze().cpu().numpy()
                # add Dirichlet noise
                noise = np.random.dirichlet([self.epsilon] * len(policy))
                policy = (1 - self.tau) * policy + self.tau * noise
                policy = policy * self.env.get_valid_actions(node.state)
                policy /= np.sum(policy)

                value = value.item()

                node = self._expand(node, policy)

            self._backpropagate(node, value)

        action_probs = np.zeros(self.env.action_space)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)

        return action_probs

    def _expand(self, node, policy):
        assert policy.sum() > 0, f"Invalid policy for node {node.state}"
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state, value, done = self.env._step(node.state, action, player=1)

                child = Node(child_state, -value, done, parent=node, action_taken=action, prior=prob)
                node.children.append(child)
        return child

    def _select(self, node: Node):
        best_child = None
        best_ucb = -np.inf

        for child in node.children:
            ucb = self._get_ucb(child)
            if ucb > best_ucb:
                best_ucb = ucb
                best_child = child
        
        return best_child
    
    def _get_ucb(self, node: Node):
        if node.visit_count == 0:
            q_value = 0
        else:
            q_value = -node.value_sum / node.visit_count
        N = node.parent.visit_count
        return q_value + self.C * node.prior * np.sqrt(N) / (1 + node.visit_count)

    def _backpropagate(self, node, value):
        node.visit_count += 1
        node.value_sum += value
        value = -value
        while node.parent is not None:
            node.parent.visit_count += 1
            node.parent.value_sum += value
            node = node.parent
            value = -value
