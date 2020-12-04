import gym

import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.sac import MlpPolicy, CnnPolicy
import os 

env = gym.make("CarRacing-v0")
policy_kwargs = dict(net_arch=dict(pi=[64, 64], qf=[400, 300]))
exp_name = "car-racing"
tensorboard_log = os.path.join("./logs", exp_name)

model = SAC(CnnPolicy, env, verbose=1, buffer_size=10000, policy_kwargs=policy_kwargs, tensorboard_log=tensorboard_log)
model.learn(total_timesteps=50000, log_interval=4)
model.save(exp_name)

del model # remove to demonstrate saving and loading

model = SAC.load(exp_name)

obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()