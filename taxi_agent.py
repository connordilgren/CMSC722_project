from collections import defaultdict
import gymnasium as gym
import numpy as np
import pickle


class TaxiAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
        q_values_path: str = None
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
            q_values_path: file path to save/load q_values
        """
        self.env = env

        self.q_values_path = q_values_path
        if q_values_path is None:
            self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        else:
            self.q_values = self.load_q_values()

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

    def load_q_values(self):
        with open(self.q_values_path, 'rb') as f:
            return defaultdict(lambda: np.zeros(self.env.action_space.n), pickle.load(f))

    def save_q_values(self, q_values_path):
        with open(q_values_path, 'wb') as f:
            pickle.dump(dict(self.q_values), f)

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
