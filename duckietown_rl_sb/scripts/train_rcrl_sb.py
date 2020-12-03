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

model = RCRLSAC(RCRLPolicy, env, verbose=1, buffer_size=10000, tensorboard_log=tensorboard_log)
model.init_replay_buffer()
# model = SAC(CnnPolicy, env, verbose=1, buffer_size=10000, tensorboard_log="./log/stable_baseline_duck_none/")

reward_log = {}
for i in range(total_timesteps // save_every):
    model.learn(total_timesteps=save_every, log_interval=4, tb_log_name="first_run")
    done = False
    total_reward = []
    obs = env.reset()
    for i in range(3):
        i_reward = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            # Perform action
            obs, reward, done, _ = env.step(action)
            i_reward += reward 
        total_reward.append(i_reward)
    
    reward_log[i] = np.mean(total_reward), np.std(total_reward)
    with open(os.path.join(tensorboard_log, "reward_log.json"), "w") as f:
        json.dump(reward_log, f)
    obs = env.reset()
    
    model.save(exp_name)

model.save(exp_name)

# del model # remove to demonstrate saving and loading

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
