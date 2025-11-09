import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
MODULE_DIR = os.path.join(ROOT_DIR, "checkpoints")
print(f"ROOT_DIR = {ROOT_DIR}")
print(f"MODULE_DIR = {MODULE_DIR}")

from stable_baselines3.common.env_checker import check_env
import gymnasium as gym
import numpy as np
from envs.projectairsim_uav_env import ProjectAirSimSmallCityEnv


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(2)  
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(3,), dtype=float)  

    def reset(self, *, seed=None, options=None):
        return (np.array([0.0, 0.0, 0.0], dtype=np.float32), {})

    def step(self, action):
        if action == 0:
            reward = 1
        else:
            reward = -1
        done = False  
        truncated = False
        info = {}
        return (np.array([0.0, 0.0, 0.0], dtype=np.float32), reward, done, truncated, info)



"""
    The sb3 provides a helper to check that your environment runs without error:
"""
def main():
    env = CustomEnv()
    check_env(env)

    env = ProjectAirSimSmallCityEnv()
    check_env(env)

if __name__ == "__main__":
    main()
