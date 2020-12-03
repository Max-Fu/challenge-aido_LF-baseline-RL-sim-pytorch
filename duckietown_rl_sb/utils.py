import random

import numpy as np
import torch


def seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# Code based on:
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

# Simple replay buffer
class ReplayBuffer(object):
    def __init__(self, max_size, additional=False): 
        # use additional to store other values, i.e. prior information
        self.storage = []
        self.max_size = max_size
        self.additional = additional 
        self.rew = []

    def __len__(self):
        return len(self.storage)

    def get_reward(self, up_to=5000): # was 1000
        if len(self.rew):
            return np.array(self.rew[:min(len(self), up_to)])
        return np.array([0])

    # Expects tuples of (state, next_state, action, reward, done)
    def add(self, state, next_state, action, reward, done, additional=None):
        if len(self.storage) < self.max_size:
            self.storage.append((state, next_state, action, reward, done, additional))
            self.rew.append(reward)
        else:
            # Remove random element in the memory beforea adding a new one
            idx = random.randrange(len(self.storage))
            self.storage.pop(idx)
            self.storage.append((state, next_state, action, reward, done, additional))
            self.rew.pop(idx)
            self.rew.append(reward)

    def sample(self, batch_size=100, flat=True):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        states, next_states, actions, rewards, dones, additionals = [], [], [], [], [], []

        for i in ind:
            state, next_state, action, reward, done, additional = self.storage[i]

            if flat:
                states.append(np.array(state, copy=False).flatten())
                next_states.append(np.array(next_state, copy=False).flatten())
            else:
                states.append(np.array(state, copy=False))
                next_states.append(np.array(next_state, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(np.array(reward, copy=False))
            dones.append(np.array(done, copy=False))
            additionals.append(np.array(additional, copy=False))
        # state_sample, action_sample, next_state_sample, reward_sample, done_sample, additional_sample if self.additional is set to true
        if self.additional:
            return {
                "state": np.stack(states),
                "next_state": np.stack(next_states),
                "action": np.stack(actions),
                "reward": np.stack(rewards).reshape(-1, 1),
                "done": np.stack(dones).reshape(-1, 1),
                "additional": np.stack(additionals),
            }
        return {
            "state": np.stack(states),
            "next_state": np.stack(next_states),
            "action": np.stack(actions),
            "reward": np.stack(rewards).reshape(-1, 1),
            "done": np.stack(dones).reshape(-1, 1),
        }


def evaluate_policy(env, policy, eval_episodes=10, max_timesteps=500):
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        step = 0
        while not done and step < max_timesteps:
            action = policy.predict(np.array(obs))
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
            step += 1

    avg_reward /= eval_episodes

    return avg_reward

def normalize(data, mean, std, eps=1e-8):
    return (data-mean)/(std+eps)

def unnormalize(data, mean, std):
    return data*std+mean

def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()