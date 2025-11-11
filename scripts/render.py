import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from stable_baselines3 import PPO
from envs.projectairsim_uav_env import ProjectAirSimSmallCityEnv
from projectairsim.utils import projectairsim_log

def main():
    env = ProjectAirSimSmallCityEnv()

    model = None
    
    for i in range(5):
        projectairsim_log().info(f"===== Episode {i + 1} start =====")
        obs, info = env.reset()
        done = truncated = False
        while not (done or truncated):
            action = [0.5, 0.5, 0.5]
            obs, reward, done, truncated, info = env.step(action)
    env.close()

if __name__ == "__main__":
    main()