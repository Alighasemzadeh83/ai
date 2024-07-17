from env import tic_tac_toe
from agent import model, visualize

env = tic_tac_toe.TicTacToe()
env.reset()
env.render()

player1_actions = [0, 4, 6, 2]
player2_actions = [1, 3, 5, 7]

winner = 0

net = model.QNetwork(in_channels=env.N, action_space=env.action_space)
net.eval()

for action1, action2 in zip(player1_actions, player2_actions):
    state, reward, done = env.step(action1)
    # env.render()
    # print(state, reward)
    policy, value = net(state.unsqueeze(0))
    print(policy, value)
    visualize.plot_policy(policy, value)
    if done:
        winner = 1
        break

    _, _, done = env.step(action2)
    # env.render()
    if done:
        winner = -1
        break

if winner == 0:
    print("Draw")
elif winner == 1:
    print("Player 1 wins")
else:
    print("Player 2 wins")