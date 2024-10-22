import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm import tqdm

from taxi_agent import TaxiAgent


def basic_plot(x, y, title, xlabel="X", ylabel="Y", save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')


def rolling_avg(x, size=100):
    avgs = [sum(x[i:i+size]) / size for i in range(len(x)-size)]
    return avgs


# hyperparameters
learning_rate = 0.01
n_episodes = 1_000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.1

env = gym.make("Taxi-v3")
# env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)

agent = TaxiAgent(
    env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)


rewards = []
episode_lens = []

for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    done = False

    episode_len = 0

    # play one episode
    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        # update the agent
        agent.update(obs, action, reward, terminated, next_obs)

        # update if the environment is done and the current obs
        done = terminated or truncated
        obs = next_obs

        episode_len += 1

    agent.decay_epsilon()

    rewards.append(reward)
    episode_lens.append(episode_len)


# save the learned Q-values
q_values_path = 'q_values.pkl'
agent.save_q_values(q_values_path)

# plot reward vs episode number
rewards_rolling_avg = rolling_avg(rewards)
skip_eps = list(range(len(rewards_rolling_avg)))
basic_plot(skip_eps[::100], rewards_rolling_avg[::100], "Episode Reward vs Episode Number", xlabel="Episode Number", ylabel="Episode Reward", save_path='.\\episode_reward.png')

# plot episode length vs episode number
episode_lens_rolling_avg = rolling_avg(episode_lens)
basic_plot(skip_eps[::100], episode_lens_rolling_avg[::100], "Episode Length vs Episode Number", xlabel="Episode Number", ylabel="Episode Length", save_path='.\\episode_length.png')

# plot TD error vs episode number
TD_error_rolling_avg = rolling_avg(agent.training_error)
skip_step = list(range(len(TD_error_rolling_avg)))
basic_plot(skip_step[::100], TD_error_rolling_avg[::100], "Temporal-Difference Error vs Episode Number", xlabel="Step Number", ylabel="TD Error", save_path='.\\episode_TD.png')
