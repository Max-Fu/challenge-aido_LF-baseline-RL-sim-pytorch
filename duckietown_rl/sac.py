import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import collections, random
from utils import normalize, unnormalize, to_numpy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Hyperparameters
lr_pi           = 0.0005
lr_q            = 0.001
init_alpha     = 0.01
gamma           = 0.98
tau             = 0.01 # for target network soft update
target_entropy = -1.0 # for automated alpha update
lr_alpha        = 0.0001  # for automated alpha update

class ActorCNN(nn.Module):
    def __init__(self, action_dim, max_action, learning_rate):
        super(ActorCNN, self).__init__()

        # ONLY TRU IN CASE OF DUCKIETOWN:
        flat_size = 32 * 6 * 6

        self.lr = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.sigm = nn.Sigmoid()

        self.conv1 = nn.Conv2d(3, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 32, 2, stride=2)
        self.conv3 = nn.Conv2d(32, 32, 2, stride=2)
        self.conv4 = nn.Conv2d(32, 32, 2, stride=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(32)

        self.dropout = nn.Dropout(.5)

        self.lin1 = nn.Linear(flat_size, 512)
        self.fc_mu = nn.Linear(512, action_dim)
        self.fc_std = nn.Linear(512, action_dim)
        self.max_action = max_action

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.log_alpha = torch.tensor(np.log(init_alpha))
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=learning_rate)


    def forward(self, x):

        x = self.bn1(self.lr(self.conv1(x)))
        x = self.bn2(self.lr(self.conv2(x)))
        x = self.bn3(self.lr(self.conv3(x)))
        x = self.bn4(self.lr(self.conv4(x)))
        try:
            x = x.view(x.size(0), -1)  # flatten
        except RuntimeError:
            x = x.reshape(x.size(0), -1)
        x = self.dropout(x)
        x = self.lr(self.lin1(x))

        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))
        dist = Normal(mu, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action)

        action[:, 0] = self.max_action * self.sigm(action[:, 0])  # because we don't want the duckie to go backwards
        action[:, 1] = self.tanh(action[:, 1])
        
        real_log_prob = log_prob - torch.log(1-torch.tanh(action).pow(2) + 1e-7)
        return action, real_log_prob

    def train_net(self, q1, q2, mini_batch):
        s, a, r, s_prime, done = mini_batch
        a_prime, log_prob = self.forward(s_prime)
        entropy = torch.mean(-self.log_alpha.detach().exp() * log_prob)

        q1_val, q2_val = q1(s,a_prime), q2(s,a_prime)
        q1_q2 = torch.cat([q1_val, q2_val], dim=1)
        min_q = torch.min(q1_q2, 1, keepdim=True)[0]

        loss = -min_q - entropy # for gradient ascent
        if torch.isnan(loss).any():
            print("NaN error! Skipped Actor")
            return torch.zeros(1), torch.zeros(1)
        self.optimizer.zero_grad()
        loss.mean().backward()
        nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()

        alpha_loss = -(self.log_alpha * (log_prob + target_entropy).detach()).mean()
        if torch.isnan(alpha_loss).any():
            print("NaN error! Skipped alpha_loss")
            return loss.mean(), torch.zeros(1)
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.log_alpha_optimizer.step()
        
        return loss.mean(), alpha_loss


class CriticCNN(nn.Module):
    def __init__(self, action_dim, learning_rate):
        super(CriticCNN, self).__init__()

        flat_size = 32 * 6 * 6

        self.lr = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.sigm = nn.Sigmoid()

        self.conv1 = nn.Conv2d(3, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 32, 2, stride=2)
        self.conv3 = nn.Conv2d(32, 32, 2, stride=2)
        self.conv4 = nn.Conv2d(32, 32, 2, stride=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(32)

        self.dropout = nn.Dropout(.5)

        self.lin1 = nn.Linear(flat_size, 256)
        self.lin2 = nn.Linear(256 + action_dim, 128)
        self.lin3 = nn.Linear(128, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, states, actions):
        x = self.bn1(self.lr(self.conv1(states)))
        x = self.bn2(self.lr(self.conv2(x)))
        x = self.bn3(self.lr(self.conv3(x)))
        x = self.bn4(self.lr(self.conv4(x)))
        try:
            x = x.view(x.size(0), -1)  # flatten
        except RuntimeError:
            x = x.reshape(x.size(0), -1)
        x = self.lr(self.lin1(x))
        x = self.lr(self.lin2(torch.cat([x, actions], 1)))  # c
        x = self.lin3(x)

        return x

    def train_net(self, target, mini_batch):
        s, a, r, s_prime, done = mini_batch
        loss = F.smooth_l1_loss(self.forward(s, a), target)
        self.optimizer.zero_grad()
        if torch.isnan(loss).any():
            print("NaN error! Skipped Critic")
            return torch.zeros(1)
        loss.mean().backward()
        nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        return loss.mean()
    
    def soft_update(self, net_target):
        for param_target, param in zip(net_target.parameters(), self.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)

def calc_target(pi, q1, q2, mini_batch):
    s, a, r, s_prime, done = mini_batch

    with torch.no_grad():
        a_prime, log_prob = pi(s_prime)
        entropy = torch.mean(-pi.log_alpha.detach().exp() * log_prob)
        q1_val, q2_val = q1(s_prime,a_prime), q2(s_prime,a_prime)
        q1_q2 = torch.cat([q1_val, q2_val], dim=1)
        min_q = torch.min(q1_q2, 1, keepdim=True)[0]
        target = r + gamma * done * (min_q + entropy)

    return target

class SAC(object):
    def __init__(self, state_dim, action_dim, max_action, learning_rate=lr_alpha, normalize_rew=True):
        super(SAC, self).__init__()

        self.state_dim = state_dim
        self.flat = False

        self.actor = ActorCNN(action_dim, max_action, learning_rate).to(device)

        self.critic_1 = CriticCNN(action_dim, learning_rate).to(device)
        self.critic_1_target = CriticCNN(action_dim, learning_rate).to(device)
        self.critic_2 = CriticCNN(action_dim, learning_rate).to(device)
        self.critic_2_target = CriticCNN(action_dim, learning_rate).to(device)

        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        self.normalize_rew = normalize_rew

    def predict(self, state):
        # just making sure the state has the correct format, otherwise the prediction doesn't work
        assert state.shape[0] == 3

        if self.flat:
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        else:
            state = torch.FloatTensor(np.expand_dims(state, axis=0)).to(device)

        state = state.detach()
        self.actor.eval()
        with torch.no_grad():
            action = to_numpy(self.actor(state)[0]).flatten()
        self.actor.train()
        return action

    def train(self, replay_buffer, iterations, batch_size=64, discount=0.99, tau=0.001):
        total_critic_1_loss = 0 
        total_critic_2_loss = 0 
        total_actor_loss = 0 
        total_alpha_loss = 0
        for it in range(iterations):

            # Sample replay buffer
            sample = replay_buffer.sample(batch_size, flat=self.flat)
            state = torch.FloatTensor(sample["state"]).to(device)
            action = torch.FloatTensor(sample["action"]).to(device)
            next_state = torch.FloatTensor(sample["next_state"]).to(device)
            done = torch.FloatTensor(1 - sample["done"]).to(device)
            reward = torch.FloatTensor(sample["reward"]).to(device)

            if self.normalize_rew:
                past_k_rewards = replay_buffer.get_reward()
                reward_mean, reward_std = np.mean(past_k_rewards), np.std(past_k_rewards)
                reward = normalize(reward, reward_mean, reward_std)

            # recombine into a minibatch 
            mini_batch = (state, action, reward, next_state, done)

            # calculate TD target
            td_target = calc_target(self.actor, self.critic_1_target, self.critic_2_target, mini_batch)

            # train q1 and q2 
            critic_loss_1 = self.critic_1.train_net(td_target, mini_batch)
            critic_loss_2 = self.critic_2.train_net(td_target, mini_batch)

            # calculate entropy 
            actor_loss, alpha_loss = self.actor.train_net(self.critic_1, self.critic_2, mini_batch)

            # update critic
            self.critic_1.soft_update(self.critic_1_target)
            self.critic_2.soft_update(self.critic_2_target)

            # summarize losses
            total_critic_1_loss += to_numpy(critic_loss_1)
            total_critic_2_loss += to_numpy(critic_loss_2)
            total_actor_loss += to_numpy(actor_loss)
            total_alpha_loss += to_numpy(alpha_loss)

        return {
            "critic_loss_1": total_critic_1_loss / iterations, 
            "critic_loss_2": total_critic_2_loss / iterations,  
            "actor_loss": total_actor_loss / iterations,
            "alpha_loss": total_alpha_loss / iterations,
        }

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '{}/{}_actor.pth'.format(directory, filename))
        torch.save(self.critic_1.state_dict(), '{}/{}_critic_1.pth'.format(directory, filename))
        torch.save(self.critic_2.state_dict(), '{}/{}_critic_2.pth'.format(directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('{}/{}_actor.pth'.format(directory, filename), map_location=device))
        self.critic_1.load_state_dict(torch.load('{}/{}_critic_1.pth'.format(directory, filename), map_location=device))
        self.critic_2.load_state_dict(torch.load('{}/{}_critic_2.pth'.format(directory, filename), map_location=device))
