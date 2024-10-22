from collections import defaultdict
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class TaxiAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            env: The training environment
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    @classmethod
    def obs_unpack(cls, obs: int) -> tuple[int, int, int, int]:
        taxi_row = obs // 100
        obs %= 100

        taxi_col = obs // 20
        obs %= 20

        passenger_loc = obs // 4
        obs %= 4

        destination = obs

        return taxi_row, taxi_col, passenger_loc, destination

    @classmethod
    def obs_pack(cls, unpacked_obs: tuple[int, int, int, int]) -> int:
        obs = unpacked_obs[0] * 100 + unpacked_obs[1] * 20 + unpacked_obs[2] * 4 + unpacked_obs[3]
        return obs

    def get_action(self, obs: int) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.

        obs is [taxi_row, taxi_col, passenger_loc, destination]
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: int,
        action: int,
        reward: float,
        terminated: bool,
        next_obs: int,
    ):
        """Updates the Q-value of an action."""
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


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
    # avgs = []
    # for i in range(len(x)-size):
    #     avg = sum(x[i:i+size]) / size
    #     avgs.append(avg)

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