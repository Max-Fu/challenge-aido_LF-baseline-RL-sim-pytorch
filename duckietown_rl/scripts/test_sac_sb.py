import numpy as np

from stable_baselines3 import SAC
# from stable_baselines3.sac import CnnPolicy
from stable_baselines3.sac import MlpPolicy
import json
from env import launch_env
import os 

total_timesteps = 2000000
save_every = 10000
exp_name = "baseline_duck"
tensorboard_log = os.path.join("./logs", exp_name)

env = launch_env(None)

model = SAC.load("baseline_duck")

obs = env.reset()

while True:
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        # Perform action
        obs, reward, done, _ = env.step(action)
        print(action, reward)
        env.render()
    
    obs = env.reset()
