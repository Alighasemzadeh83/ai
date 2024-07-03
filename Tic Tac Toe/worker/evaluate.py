"""
Play with human player
"""

import os
import torch
import numpy as np
from env.tic_tac_toe import TicTacToe
from agent.agent import Agent

def play_game():
    env = TicTacToe()
    agent = Agent(9, 9)
    files = os.listdir("models")
    files.sort()
    agent.model.load_state_dict(torch.load(f"models/{files[-1]}"))
    env.reset()
    state = env.board
    done = False

    while not done:
        env.render()
        action = int(input("Enter action: "))
        next_state, reward, done = env.step(action)
        if done:
            break

        state = next_state
        action = agent.act(state, eps=0.0)
        next_state, reward, done = env.step(action)
        state = next_state
