import re

import gymnasium as gym
import pandas as pd
from tqdm import tqdm

from taxi_agent_monolithic import TaxiAgentMonolithic


###############################################################################
# Run the problem

q_values_path = r'q_values\monolithic\lr0.1_ed0.0002_df0.99_q_values_1000_eps.pkl'

df = pd.DataFrame({
    'learning_rate': [],
    'epsilon_decay': [],
    'discount_factor': [],
    'num_steps': []
    })

# static variables
n_episodes = 1000

env = gym.make("Taxi-v3")

agent = TaxiAgentMonolithic(
    env,
    learning_rate=0,
    initial_epsilon=0.1,
    epsilon_decay=0,
    final_epsilon=0,
    discount_factor=0,
    q_values_path=q_values_path
)

for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    done = False

    episode_len = 0
    agent.num_steps = 0

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

    new_row = pd.DataFrame({
        'learning_rate': [float(re.search(r'lr([0-9.]+)_', q_values_path).group(1))],
        'epsilon_decay': [float(re.search(r'ed([0-9.]+)_', q_values_path).group(1))],
        'discount_factor': [float(re.search(r'df([0-9.]+)_', q_values_path).group(1))],
        'num_steps': int(agent.num_steps)
    })
    df = pd.concat([df, new_row], ignore_index=True)

# save results
with open('eval_results/eval_results_monolithic_1000_eps.pkl', 'wb') as f:
    df.to_pickle(f)
