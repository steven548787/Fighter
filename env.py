import gym
import numpy as np
from game import Game  # 假设 Game 类封装了游戏核心逻辑

class CosmicHeatEnv(gym.Env):
    def __init__(self):
        super(CosmicHeatEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(3)  # 3种动作：向上、向下、不动
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        self.game = Game()

    def reset(self):
        self.game.reset()
        return self._get_obs()

    def step(self, action):
        self.game.step(action)
        state = self._get_obs()
        reward = self.game.score
        done = self.game.is_over()
        return state, reward, done, {}

    def render(self, mode='human'):
        self.game.render()

    def _get_obs(self):
        player_pos = self.game.player.get_position()
        obstacle_pos = self.game.get_obstacle_positions()
        return np.array([player_pos[0], player_pos[1], obstacle_pos[0], obstacle_pos[1]])
