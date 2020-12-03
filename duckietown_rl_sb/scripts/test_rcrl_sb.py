import numpy as np
from rcrl import RCRLSAC
# from stable_baselines3.sac import CnnPolicy
from rcrl_policy import RCRLPolicy
import json
from env import launch_env
import os 

total_timesteps = 2000000
save_every = 10000
exp_name = "RCRL_duck"
tensorboard_log = os.path.join("./logs", exp_name)
prior_dim = 2

env = launch_env(None, prior_dim)

model = RCRLSAC.load(exp_name)

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
