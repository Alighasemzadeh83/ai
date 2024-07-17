from env import tic_tac_toe
from agent import net, visualize
from worker.self_play import self_play


H = {
    "num_games": 1,
    "num_simulations": 50,
    "C": 1.0
}

env = tic_tac_toe.TicTacToe()
model = net.QNetwork(in_channels=3, action_space=env.action_space)

self_play(model, env, num_games=H["num_games"], num_simulations=H["num_simulations"], C=H["C"])