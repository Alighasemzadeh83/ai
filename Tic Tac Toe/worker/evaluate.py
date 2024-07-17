import matplotlib.pyplot as plt

def manual_play(env, agent, human_first=True):
    """
    Play against the agent
    """
    state = env.reset()
    done = False
    human_turn = human_first

    while not done:
        if human_turn:
            action = int(input("Enter action: "))
        else:
            policy, value = agent.get_policy_value(state)
            # env.render(policy, value)
            plot_policy(policy, value)
            action = policy.argmax()
        state, _, done = env.step(action)
        env.render()
        human_turn = not human_turn



def plot_policy(policy, value):
    """
    Plot two subplots: policy and value

    policy plot should be a horizontal bar plot with probabilities for each action
    value plot should be a vertical bar plot with the value of the state (between -1 and 1)
    """

    fig, axs = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={"width_ratios": [3, 1]})

    axs[0].barh(range(len(policy)), policy, color="skyblue")
    axs[0].set_xlim(0, 1)
    axs[0].set_yticks(range(len(policy)))
    axs[0].set_yticklabels(range(len(policy)))
    axs[0].set_title(f"Best action: {policy.argmax()} with probability {policy.max():.2f}")

    # y-axis range should be between -1 and 1
    axs[1].bar(range(1), value, color="salmon", align="center")
    axs[1].set_ylim(-1, 1)
    axs[1].set_xticks(range(1))
    axs[1].set_xticklabels(["Value"])
    axs[1].set_title(f"Value: {value:.2f}")

    plt.show()


