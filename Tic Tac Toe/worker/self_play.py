"""
Generate experiences based on self-play

This class is used to generate experiences based on self-play. It takes an environment and an agent as input and generates experiences based on the agent playing against itself.

Each experience is in format (state, action, reward, next_state, done). The reward is 10 if the agent wins, -10 if the agent loses, and 0 if it is a draw.

The nueral net for both players is the same, so the agent is passed as an input twice.
"""

from env.tic_tac_toe import TicTacToe
from agent.agent import Agent

import numpy as np
from datetime import datetime
import copy

class SelfPlay:
    def __init__(self, env=None, agent=None):
        self.env = env if env else TicTacToe()
        self.player1 = agent if agent else Agent(9, 9)
        self.player2 = Agent(9, 9)
        self.player2.model.load_state_dict(copy.deepcopy(self.player1.model.state_dict()))
    
    def play_game(self):
        self.env.reset()
        state = self.env.board
        done = False
        experiences = []
        
        while not done:
            action1 = self.player1.act(state, eps=0.0)
            next_state, reward, done = self.env.step(action1)
            experiences.append((state, action1, reward, next_state, done))
            if done:
                break
            
            action2 = self.player2.act(next_state, eps=0.0)
            next_state, reward, done = self.env.step(action2)
            experiences.append((state, action2, reward, next_state, done))
            state = next_state
        
        return experiences
    
    def generate_experiences(self, num_games):
        all_experiences = []
        for i in range(num_games):
            experiences = self.play_game()
            all_experiences.extend(experiences)
        
        directory = f"experiences/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        np.save(directory, all_experiences)

        return experiences
