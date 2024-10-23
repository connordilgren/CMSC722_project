import gymnasium as gym


class NavigateTaxiEnv:
    def __init__(
        self,
        env: gym.Env,
        goal_square_color: str
    ):
        self.env = env
        self.goal_square_color = goal_square_color

        self.color_to_coord = {
            'red':    (0, 0),
            'green':  (0, 5),
            'yellow': (4, 0),
            'blue':   (4, 3)
        }

        self.color_enumerated = {
            'red':    0,
            'green':  1,
            'yellow': 2,
            'blue':   3
        }

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
    def obs_pack(cls, obs_unpacked: tuple[int, int, int, int]) -> int:
        obs = obs_unpacked[0] * 100 + obs_unpacked[1] * 20 + obs_unpacked[2] * 4 + obs_unpacked[3]
        return obs

    def get_obs_simple(self, obs_env):
        taxi_row, taxi_col, _, _ = self.obs_unpack(obs_env)
        next_obs_simple = (taxi_row, taxi_col, self.color_enumerated[self.goal_square_color])
        return next_obs_simple

    def reset(self):
        obs_env, info = self.env.reset()
        obs_simple = self.get_obs_simple(obs_env)
        return obs_simple, info

    def step(self, action):
        next_obs_env, reward, terminated, truncated, info = self.env.step(action)
        next_obs_simple = self.get_obs_simple(next_obs_env)

        if (next_obs_simple[0], next_obs_simple[1]) == self.color_to_coord[self.goal_square_color]:
            reward = 20
            terminated = True

        return next_obs_simple, reward, terminated, truncated, info
