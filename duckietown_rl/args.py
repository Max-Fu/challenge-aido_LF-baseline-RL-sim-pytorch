import argparse
import sys


def get_args_train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=200, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=1e4, type=int)  # How many time steps purely random policy is run for
    parser.add_argument("--eval_freq", default=5e3, type=float)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=2.5e6, type=float)  # Max time steps to run environment for
    parser.add_argument("--save_models", action="store_true", default=True)  # Whether or not models are saved
    parser.add_argument("--expl_noise", default=0.1, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=32, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--env_timesteps", default=500, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--replay_buffer_max_size", default=10000, type=int)  # Maximum number of steps to keep in the replay buffer

    # RCRL Args
    parser.add_argument("--lr_actor", default=1e-4, type=float) # learning rate of actor (only for RCRL)
    parser.add_argument("--lr_critic", default=1e-3, type=float) # learning rate of critic (only for RCRL)
    parser.add_argument("--lr_prior", default=1e-4, type=float) # learning rate of prior (only for RCRL)
    parser.add_argument("--model-dir", type=str, default="pytorch_models")
    parser.add_argument("--rcrl", action="store_true", default=False)
    parser.add_argument("--dist_param", default=1.0, type=float) # when calculating possibility of collision, uses exp(-alpha * k)
    parser.add_argument("--env_name", required=False, default=None, type=str) # 'Duckietown-loop_pedestrians-v0'
    parser.add_argument("--new_size", required=False, default=(64, 64, 3), type=tuple) # default is (120, 160, 3)
    
    return parser.parse_args()

def get_args_test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=123, type=int)  # Inform the test what seed was used in training
    parser.add_argument("--experiment", default=2, type=int)
    
    return parser.parse_args()
