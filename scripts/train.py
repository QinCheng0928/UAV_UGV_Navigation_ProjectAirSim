import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
MODULE_DIR = os.path.join(ROOT_DIR, "checkpoints")
print(f"ROOT_DIR = {ROOT_DIR}")
print(f"MODULE_DIR = {MODULE_DIR}")

import logging
from datetime import datetime
from stable_baselines3 import PPO
from envs.projectairsim_uav_env import ProjectAirSimSmallCityEnv
from projectairsim.utils import projectairsim_log
projectairsim_log().setLevel(logging.WARNING)

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs(os.path.join(MODULE_DIR, current_time), exist_ok=True)

def main():
    env = ProjectAirSimSmallCityEnv()
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=(dict(pi=[128, 256], vf=[256, 128]))),
        n_steps=128,
        batch_size=64,
        n_epochs=10,
        learning_rate=5e-4,
        gamma=0.99,
        verbose=2,
        device="cpu",
        tensorboard_log=os.path.join(MODULE_DIR, current_time)
    )
    model.learn(total_timesteps=int(1e4))
    
    model_name = f"ppo_smallcity_{current_time}"
    save_path = os.path.join(MODULE_DIR, current_time, model_name)
    model.save(save_path)
    env.close()    
    
if __name__ == "__main__":
    main()