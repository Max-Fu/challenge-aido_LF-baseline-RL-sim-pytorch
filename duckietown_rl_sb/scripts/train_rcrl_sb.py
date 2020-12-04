import numpy as np
from rcrl import RCRLSAC
# from stable_baselines3.sac import CnnPolicy
from rcrl_policy import RCRLPolicy, CnnPolicy
import json
from env import launch_env
import os 

total_timesteps = 100000
save_every = 10000
exp_name = "RCRL_duck"
tensorboard_log = os.path.join("./logs", exp_name)
prior_dim = 2

env = launch_env(None, prior_dim)
policy_kwargs = dict(net_arch=dict(pi=[64, 64], qf=[400, 300]))
model = RCRLSAC(CnnPolicy, env, verbose=1, buffer_size=10000, policy_kwargs=policy_kwargs, tensorboard_log=tensorboard_log)
model.init_replay_buffer()

reward_log = {}
for i in range(total_timesteps // save_every):
    model.learn(total_timesteps=save_every, log_interval=4, reset_num_timesteps=False)
    
    print("start evaluation")
    done = False
    total_reward = []
    obs = env.reset()
    for i in range(3):
        i_reward = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            # Perform action
            print(action)
            obs, reward, done, _ = env.step(action)
            i_reward += reward 
        total_reward.append(i_reward)
    
    print("Iteration {}".format(i))
    print("Evaluation rewards: ", total_reward)
    reward_log[i] = np.mean(total_reward), np.std(total_reward)
    print("Evaluation statistics: ", reward_log[i])
    with open(os.path.join(tensorboard_log, "reward_log.json"), "wt") as f:
        json.dump(reward_log, f)
    obs = env.reset()
    
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
