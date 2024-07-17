from env import tic_tac_toe
from agent import net
from agent.agent import Agent
from worker.self_play import self_play
from worker.optimize import train, load_best_model
from worker.evaluate import manual_play


H = {
    "lr": 0.001,
    "num_epochs": 10,
    "batch_size": 32,
    "num_games": 100,
    "num_simulations": 500,
    "C": 1.0
}

env = tic_tac_toe.TicTacToe()
model = net.QNetwork(in_channels=3, action_space=env.action_space)
model = load_best_model(model)

### Self Play
# self_play(model, env, num_games=H["num_games"], num_simulations=H["num_simulations"], C=H["C"])

### Train the model
# train(model, transform=env.transform, lr=H["lr"], num_epochs=H["num_epochs"], batch_size=H["batch_size"])

### Manual Play
agent = Agent(model, env.transform)
manual_play(env=env, agent=agent, human_first=True)

