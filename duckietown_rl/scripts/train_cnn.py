import pickle
import random
import resource
import gym_duckietown
import numpy as np
import torch
import gym
import os

from args import get_args_train
from ddpg import DDPG
from rcrl import RCRL
from utils import seed, evaluate_policy, ReplayBuffer
from wrappers import NormalizeWrapper, ImgWrapper, \
    DtRewardWrapper, ActionWrapper, ResizeWrapper, SteeringToWheelVelWrapper
from env import launch_env
from gym_duckietown.simulator import AGENT_SAFETY_RAD
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import datetime
import hashlib

POSITION_THRESHOLD = 0.04
REF_VELOCITY = 0.7
FOLLOWING_DISTANCE = 0.24
AGENT_SAFETY_GAIN = 1.15

def train(args):
    if args.rcrl:
        policy_name = "RCRL"
    else:
        policy_name = "DDPG"

    print(f"Using {'cuda' if torch.cuda.is_available() else 'cpu'}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    time_id = hashlib.sha1(str(datetime.datetime.now()).encode('utf-8')).hexdigest()[:10]

    folder_name = "{}_{}_{}".format(
        policy_name,
        str(args.seed),
        time_id,
    )
    file_name = folder_name

    env = launch_env(args.env_name)

    # Set seeds
    seed(args.seed)

    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])


    # Initialize policy
    if args.rcrl:
        policy = RCRL(state_dim, action_dim, max_action, prior_dim=1, lr_actor=args.lr_actor, lr_critic=args.lr_critic, lr_prior=args.lr_prior)
    else: 
        policy = DDPG(state_dim, action_dim, max_action, net_type="cnn")

    replay_buffer = ReplayBuffer(args.replay_buffer_max_size, additional=args.rcrl)

    # Evaluate untrained policy
    evaluations= [evaluate_policy(env, policy)]

    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = True
    episode_reward = None
    env_counter = 0
    results_folder_path = './results/{}'.format(folder_name)
    os.makedirs(results_folder_path, exist_ok=True)
    print("Results saved to " + results_folder_path)
    if args.save_models:
        model_folder_path = os.path.join(args.model_dir, folder_name)
        os.makedirs(model_folder_path, exist_ok=True)
        print("Model saved to " + model_folder_path)
    writer = SummaryWriter(log_dir=results_folder_path)

    while total_timesteps < args.max_timesteps:

        if done:
            print(f"Done @ {total_timesteps}")

            if total_timesteps != 0:
                print("Replay buffer length is ", len(replay_buffer.storage))
                print(("Total T: %d Episode Num: %d Episode T: %d Reward: %f") % (
                    total_timesteps, episode_num, episode_timesteps, episode_reward))
                losses = policy.train(replay_buffer, episode_timesteps, args.batch_size, args.discount, args.tau)
                for tag, val in losses.items():
                    writer.add_scalar('loss/'+tag, val, total_timesteps)
                print("Tensorboard logged to  " + results_folder_path)
            # Evaluate episode
            if timesteps_since_eval >= args.eval_freq:
                timesteps_since_eval %= args.eval_freq
                evaluations.append(evaluate_policy(env, policy))
                print("rewards at time {}: {}".format(total_timesteps, evaluations[-1]))
                # Write rewards to tensorboard 
                writer.add_scalar('rewards', evaluations[-1], total_timesteps)
                print("Saving tensorboard log to {}".format(results_folder_path))
                if args.save_models:
                    policy.save(file_name, directory=model_folder_path)
                    print("Model saved to " + model_folder_path)
                np.savez(os.path.join(results_folder_path, "{}.npz".format(file_name)), evaluations)

            # Reset environment
            env_counter += 1
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Select action randomly or according to policy
        if total_timesteps < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = policy.predict(np.array(obs))
            if args.expl_noise != 0:
                action = (action + np.random.normal(
                    0,
                    args.expl_noise,
                    size=env.action_space.shape[0])
                        ).clip(env.action_space.low, env.action_space.high)

        if args.rcrl: 
            current_world_objects = env.objects
            obj_distances = []
            for obj in current_world_objects:
                if not obj.static:
                    obj_safe_dist = abs(
                        obj.proximity(env.cur_pos, AGENT_SAFETY_RAD * AGENT_SAFETY_GAIN, true_safety_dist=True)
                    )
                    obj_distances.append(obj_safe_dist)
            min_dist = min(obj_distances)
            # reduce variance by using exponential decay
            exp_neg_min_dist = np.exp(-args.dist_param * min_dist)

        # Perform action
        new_obs, reward, done, _ = env.step(action)
        if action[0] < 0.001:   #Penalise slow actions: helps the bot to figure out that going straight > turning in circles
            reward = 0

        if episode_timesteps >= args.env_timesteps:
            done = True

        done_bool = 0 if episode_timesteps + 1 == args.env_timesteps else float(done)
        episode_reward += reward

        if args.rcrl: 
            # Store data in replay buffer
            replay_buffer.add(obs, new_obs, action, reward, done_bool, exp_neg_min_dist)
        else: 
            # Store data in replay buffer
            replay_buffer.add(obs, new_obs, action, reward, done_bool)

        obs = new_obs

        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1

    # Final evaluation
    evaluations.append(evaluate_policy(env, policy))

    if args.save_models:
        policy.save(file_name, directory=model_folder_path)
    np.savez(os.path.join(results_folder_path, "{}.npz".format(file_name)), evaluations)

if __name__ == "__main__":
    args = get_args_train()
    train(args)