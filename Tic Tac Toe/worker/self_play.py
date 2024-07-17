"""
Generate experiences based on self-play.

This class is used to generate experiences based on self-play.

Each experience is in format (state, action, value). The value is 1 if the agent wins, -1 if the agent loses, and 0 if it is a draw.
"""

import os
import datetime
import numpy as np

def self_play(model, env, num_games):
    """
    Plays multiple games and save in file.
    """

    states = []
    policies = []
    values = []

    for _ in range(num_games):
        state, policy, value = __self_play(model, env)
        states.append(state)
        policies.append(policy)
        values.append(value)

    states = np.concatenate(states)
    policies = np.concatenate(policies)
    values = np.concatenate(values)

    # save the experiences
    os.makedirs("experiences", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    os.makedirs(f"experiences/{timestamp}")
    np.savez(f"experiences/{timestamp}/experiences.npz", states=states, policies=policies, values=values)

def __self_play(model, env):
    """
    Plays one game and returns the experiences.
    """

    memory = []
    mcts = MonteCarloTreeSearch(model)
    state = env.reset()
    next_state = None
    done = False
    reward = None

    while not done:
        policy = mcts.search()
        action = np.random.choice(env.action_space, p=policy)
        next_state, reward, done = env.step(action)
        memory.append((state, policy, reward))
        state = next_state
    
    states = np.array([state for state, _, _ in memory], dtype=np.int8)
    policies = np.array([policy for _, policy, _ in memory], dtype=np.float32)
    values = np.full(len(memory), reward, dtype=np.int8)
    # set value for loser to -reward
    values[len(memory)%2::2] = -reward

    return states, policies, values



class MonteCarloTreeSearch:
    def __init__(self, model):
        pass

    def search(self):
        """
        Returns the policy based on the MCTS
        """
        pass