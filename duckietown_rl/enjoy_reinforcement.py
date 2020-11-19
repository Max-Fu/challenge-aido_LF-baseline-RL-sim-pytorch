import ast
import argparse
import logging

import os
import numpy as np

import random
import resource
import gym_duckietown
import torch
import gym

# Duckietown Specific
from ddpg import DDPG
from rcrl import RCRL
from env import launch_env
from wrappers import NormalizeWrapper, ImgWrapper, DtRewardWrapper, ActionWrapper, ResizeWrapper

def _enjoy(args):
    # Launch the env with our helper function
    env = launch_env(args.env_name)
    print("Initialized environment")

    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Initialize policy
    if args.rcrl:
        policy = RCRL(state_dim, action_dim, max_action, prior_dim=1, lr_actor=args.lr_actor, lr_critic=args.lr_critic, lr_prior=args.lr_prior)
    else:
        policy = DDPG(state_dim, action_dim, max_action, net_type="cnn")
    if args.rcrl: 
        fn = "rcrl"
    else:
        fn = "ddpg"
    policy.load(filename=args.folder_hash, directory=os.path.join("pytorch_models", args.folder_hash))

    obs = env.reset()
    done = False

    while True:
        while not done:
            action = policy.predict(np.array(obs))
            # Perform action
            obs, reward, done, _ = env.step(action)
            env.render()
        done = False
        obs = env.reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rcrl", action="store_true", default=False)
    parser.add_argument("--lr_actor", default=1e-4, type=float) # learning rate of actor (only for RCRL)
    parser.add_argument("--lr_critic", default=1e-3, type=float) # learning rate of critic (only for RCRL)
    parser.add_argument("--lr_prior", default=1e-4, type=float) # learning rate of prior (only for RCRL)
    parser.add_argument("--folder_hash", required=True, type=str)
    parser.add_argument("--env_name", required=False, default='Duckietown-loop_pedestrians-v0', type=str)
    _enjoy(parser.parse_args())
