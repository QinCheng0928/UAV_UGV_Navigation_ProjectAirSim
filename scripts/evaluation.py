import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
MODULE_DIR = os.path.join(ROOT_DIR, "checkpoints", "20251112_184349")

from stable_baselines3 import PPO
from envs.projectairsim_uav_env import ProjectAirSimSmallCityEnv

def main():
    env = ProjectAirSimSmallCityEnv()
    
    model_name = "ppo_smallcity_20251112_184349.zip"
    model = PPO.load(os.path.join(MODULE_DIR, model_name))
    
    options={
        "display": True,
        "save": True,
    }

    for _ in range(1):
        obs, info = env.reset(options = options)
        done = truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs)
            print(action)
            obs, reward, done, truncated, info = env.step(action)
    if options["save"]:
        env.close_all_videos()
    env.close()


if __name__ == "__main__":
    main()