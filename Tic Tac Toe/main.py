from env import connect4, tic_tac_toe
from agent import net
from agent.agent import Agent
from worker.self_play import self_play
from worker.optimize import train, load_best_model
from worker.evaluate import manual_play

import argparse

H = {
    "lr": 0.001,
    "num_epochs": 10,
    "batch_size": 32,
    "num_games": 100,
    "num_simulations": 500,
    "C": 1.0,
    "tau": 0,
    "epsilon": 0.25
}

# env = connect4.Connect4()
env = tic_tac_toe.TicTacToe()
model = net.QNetwork(in_channels=3, state_space=env.state_space, action_space=env.action_space)
model = load_best_model(model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", help="Mode of the program", required=True)
    args = parser.parse_args()

    if args.mode == "self_play":
        self_play(model, env, H["num_games"], H["num_simulations"], H["C"], H["tau"], H["epsilon"])
    elif args.mode == "train":
        train(model, env.transform, H["lr"], H["num_epochs"], H["batch_size"])
    elif args.mode == "manual_play":
        agent = Agent(model, env.transform)
        manual_play(env, agent, human_first=False)
    else:
        print("Invalid mode")