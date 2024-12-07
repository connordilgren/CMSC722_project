from collections import defaultdict
import os
import shutil

import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm import tqdm

from taxi_agent_monolithic import TaxiAgentMonolithic


def basic_plot(x, y, title, xlabel="X", ylabel="Y", save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()

def rolling_avg(x, size=100):
    avgs = [sum(x[i:i+size]) / size for i in range(len(x)-size)]
    return avgs


# static variables
n_episodes = 1_000 * 8
start_epsilon = 1.0
final_epsilon = 0.1

# independent variables
learning_rate = 0.1
discount_factor = 0.99
epsilon_decay = 0.0002

env = gym.make("Taxi-v3", render_mode='human')
rewards = []
episode_lens = []

agent = TaxiAgentMonolithic(
    env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
    discount_factor=discount_factor
)

for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    done = False

    episode_len = 0
    env.render()
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

# set up figure save directory
trial_name = f'lr{learning_rate}_ed{epsilon_decay}_df{discount_factor}'.replace('.', '_')
save_dir = f".\\training_plots\\monolithic\\{trial_name}"

# if not os.path.exists(".\\training_plots"):
#     os.makedirs(".\\training_plots")

# if os.path.exists(save_dir):
#     shutil.rmtree(save_dir)
# os.makedirs(save_dir)

# plot reward vs episode number
rewards_rolling_avg = rolling_avg(rewards, size=20)
skip_eps = list(range(len(rewards_rolling_avg)))
basic_plot(skip_eps,
            rewards_rolling_avg,
            "Episode Reward vs Episode Number",
            xlabel="Episode Number",
            ylabel="Episode Reward",
            save_path=f'{save_dir}\\episode_reward_{n_episodes}_eps.png')

# plot episode length vs episode number
episode_lens_rolling_avg = rolling_avg(episode_lens, size=20)
basic_plot(skip_eps,
            episode_lens_rolling_avg,
            "Episode Length vs Episode Number",
            xlabel="Episode Number",
            ylabel="Episode Length",
            save_path=f'{save_dir}\\episode_length_{n_episodes}_eps.png')

# plot TD error vs step number
TD_error_rolling_avg = rolling_avg(agent.training_error, size=100)
skip_step = list(range(len(TD_error_rolling_avg)))
basic_plot(skip_step,
            TD_error_rolling_avg,
            "Temporal-Difference Error vs Step Number",
            xlabel="Step Number",
            ylabel="TD Error",
            save_path=f'{save_dir}\\episode_TD_{n_episodes}_eps.png')

# # save the learned Q-values
# q_values_path = f'.\\q_values\\monolithic\\lr{learning_rate}_ed{epsilon_decay}_df{discount_factor}_q_values_{n_episodes}_eps.pkl'
# agent.save_q_values_gs(q_values_path)
