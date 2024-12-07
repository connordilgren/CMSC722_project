from collections import defaultdict
import numpy as np
import pickle


class TaxiAgentMonolithic:
    def __init__(
        self,
        env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float,
        q_values_path: str = None,
        num_steps: int = 0
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        q_values is a dict whose keys is an obs (an int describing the env 
            state) and values are the Q values of each action

        Args:
            env: The training environment
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
            q_values_path: file path to save/load q_values_gs
        """
        self.env = env

        self.q_values_path = q_values_path
        if q_values_path is None:
            self.q_values = defaultdict(self.initialize_q_values)
        else:
            self.q_values = self.load_q_values()

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

        self.num_steps = num_steps

    def initialize_q_values(self):
        return np.zeros(self.env.action_space.n)

    def load_q_values(self):
        with open(self.q_values_path, 'rb') as f:
            return pickle.load(f)

    def save_q_values_gs(self, q_values_path):
        with open(q_values_path, 'wb') as f:
            pickle.dump(self.q_values, f)

    def get_action(self, obs: int) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.

        obs is an int() that encodes the corresponding state, calculated by 
            ((taxi_row * 5 + taxi_col) * 5 + passenger_location) * 4 + 
            destination (from Gymnasium Taxi documentation)
        """
        self.num_steps += 1
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
