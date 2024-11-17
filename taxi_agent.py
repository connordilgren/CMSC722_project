from collections import defaultdict
import numpy as np
import pickle
from taxi_env import NavigateTaxiEnv


class TaxiAgent:
    def __init__(
        self,
        nav_env: NavigateTaxiEnv,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float,
        q_values_gs_path: str = None,
        num_steps: int = 0
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        q_values_gs is a dict with keys in ['red', 'green', 'yellow', 'blue'] 
            which represent the destination color and values which are 
            q_values
        q_values is a dict with keys in (taxi_row, taxi_col, destination) 
            and values are q values

        Args:
            nav_env: The training environment
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
            q_values_gs_path: file path to save/load q_values_gs
        """
        self.nav_env = nav_env

        self.q_values_gs_path = q_values_gs_path
        if q_values_gs_path is None:
            self.q_values_gs = defaultdict(self.initialize_q_values_gs)
        else:
            self.q_values_gs = self.load_q_values_gs()

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = defaultdict(list)

        self.num_steps = num_steps

    def initialize_q_values_gs(self):
        return defaultdict(self.initialize_q_values)

    def initialize_q_values(self):
        return np.zeros(self.nav_env.env.action_space.n)

    def load_q_values_gs(self):
        with open(self.q_values_gs_path, 'rb') as f:
            return pickle.load(f)

    def save_q_values_gs(self, q_values_gs_path):
        with open(q_values_gs_path, 'wb') as f:
            pickle.dump(self.q_values_gs, f)

    def get_action(self, obs: tuple, goal_square_color: str) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.

        obs is (taxi_row, taxi_col)
        goal_square_color is the square the taxi wants to get to
        """
        self.num_steps += 1
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return self.nav_env.env.action_space.sample()
        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.q_values_gs[goal_square_color][obs]))

    def update(
        self,
        obs: int,
        action: int,
        reward: float,
        terminated: bool,
        next_obs: int,
        goal_square_color: str,
    ):
        """Updates the Q-value of an action."""
        future_q_value = (not terminated) * np.max(self.q_values_gs[goal_square_color][next_obs])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values_gs[goal_square_color][obs][action]
        )

        self.q_values_gs[goal_square_color][obs][action] = (
            self.q_values_gs[goal_square_color][obs][action] + self.lr * temporal_difference
        )
        self.training_error[goal_square_color].append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
