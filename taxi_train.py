from collections import defaultdict
import os
import shutil

import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm import tqdm

from taxi_agent import TaxiAgent
from taxi_env import NavigateTaxiEnv


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
n_episodes = 5_000
start_epsilon = 1.0
final_epsilon = 0.1

# independent variables
learning_rates = [0.001, 0.01, 0.1]
discount_factors = [0.50, 0.95, 0.99]
# discount_factors = [0.95]
epsilon_decays = [start_epsilon / (n_episodes / x) for x in [1, 2, 4]]  # reduce the exploration over time
# epsilon_decays = [start_epsilon / (n_episodes / x) for x in [2]]  # reduce the exploration over time

env = gym.make("Taxi-v3")
rewards = defaultdict(list)
episode_lens = defaultdict(list)

trials = []
for learning_rate in learning_rates:
    for discount_factor in discount_factors:
        for epsilon_decay in epsilon_decays:
            trial = {
                'learning_rate': learning_rate,
                'discount_factor': discount_factor,
                'epsilon_decay': epsilon_decay
            }
            trials.append(trial)

for trial in trials:
    print(f'Trial: lr{trial['learning_rate']}_ed{trial['epsilon_decay']}_df{trial['discount_factor']}')

    nav_env = NavigateTaxiEnv(env, None)
    agent = TaxiAgent(
        nav_env,
        learning_rate=trial['learning_rate'],
        initial_epsilon=start_epsilon,
        epsilon_decay=trial['epsilon_decay'],
        final_epsilon=final_epsilon,
        discount_factor=trial['discount_factor']
    )

    for goal_square_color in ['red', 'green', 'yellow', 'blue']:
        nav_env.goal_square_color = goal_square_color
        agent.epsilon = start_epsilon

        for episode in tqdm(range(n_episodes)):
            obs, info, _, _ = nav_env.reset()
            done = False

            episode_len = 0

            # play one episode
            while not done:
                action = agent.get_action(obs, goal_square_color)
                next_obs, reward, terminated, truncated, info = nav_env.step(action)

                # update the agent
                agent.update(obs, action, reward, terminated, next_obs, goal_square_color)

                # update if the environment is done and the current obs
                done = terminated or truncated
                obs = next_obs

                episode_len += 1

            agent.decay_epsilon()

            rewards[goal_square_color].append(reward)
            episode_lens[goal_square_color].append(episode_len)

        # set up figure save directory
        trial_name = f'lr{trial['learning_rate']}_ed{trial['epsilon_decay']}_df{trial['discount_factor']}'.replace('.', '_')
        save_dir = f".\\training_plots\\{trial_name}"

        if not os.path.exists(".\\training_plots"):
            os.makedirs(".\\training_plots")

        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir)

        # plot reward vs episode number
        rewards_rolling_avg = rolling_avg(rewards[goal_square_color], size=10)
        skip_eps = list(range(len(rewards_rolling_avg)))
        basic_plot(skip_eps[::100],
                   rewards_rolling_avg[::100],
                   "Episode Reward vs Episode Number",
                   xlabel="Episode Number",
                   ylabel="Episode Reward",
                   save_path=f'{save_dir}\\{goal_square_color}_episode_reward.png')

        # plot episode length vs episode number
        episode_lens_rolling_avg = rolling_avg(episode_lens[goal_square_color], size=10)
        basic_plot(skip_eps[::100],
                   episode_lens_rolling_avg[::100],
                   "Episode Length vs Episode Number",
                   xlabel="Episode Number",
                   ylabel="Episode Length",
                   save_path=f'{save_dir}\\{goal_square_color}_episode_length.png')

        # plot TD error vs step number
        TD_error_rolling_avg = rolling_avg(agent.training_error[goal_square_color], size=10)
        skip_step = list(range(len(TD_error_rolling_avg)))
        basic_plot(skip_step[::100],
                   TD_error_rolling_avg[::100],
                   "Temporal-Difference Error vs Step Number",
                   xlabel="Step Number",
                   ylabel="TD Error",
                   save_path=f'{save_dir}\\{goal_square_color}_episode_TD.png')

    # save the learned Q-values
    q_values_gs_path = f'.\\q_values\\lr{trial['learning_rate']}_ed{trial['epsilon_decay']}_df{trial['discount_factor']}_q_values_gs.pkl'
    agent.save_q_values_gs(q_values_gs_path)
