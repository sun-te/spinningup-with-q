import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from ipdb import set_trace as tt

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


LOG_STD_MAX = 2
LOG_STD_MIN = -20


class Discriminator(nn.Module):
    def __init__(self, obs_dim, act_dim, activation):
        super().__init__()
        self.net = mlp([obs_dim + act_dim + 1, 256, 256, 1], activation, torch.nn.Identity)

    def forward(self, obs, act, q):
        net_out = self.net(torch.cat([obs, act, q.view(obs.shape[0], -1)], dim=1))
        softmax = torch.softmax(net_out, dim=0)
        return softmax

class SquashedGaussianMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding 
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290) 
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi


class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):

        q = self.q(torch.cat([obs, act], dim=-1))

        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.d = Discriminator(obs_dim, act_dim, activation)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}


class DemoBuffer:
    def __init__(self, size=100):
        self.size = int(size)
        self.obs_buf = []
        self.obs2_buf = []
        self.act_buf = []
        self.rew_buf = []
        self.done_buf = []
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        if self.ptr >= self.max_size:
            self.ptr += 1
            idx = int((self.ptr) % self.max_size)
            self.obs_buf[idx] = obs
            self.obs2_buf[idx] = next_obs
            self.act_buf[idx] = act
            self.rew_buf[idx] = rew
            self.done_buf[idx] = done
        else:
            self.size += 1
            self.ptr += 1
            self.obs2_buf.append(next_obs)
            self.obs_buf.append(obs)
            self.act_buf.append(act)
            self.rew_buf.append(rew)
            self.done_buf.append(done)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}

    def save(self, output_file):
        key = ['obs_buf', 'obs2_buf', 'act_buf', 'rew_buf', 'done_buf']
        data = [self.obs_buf, self.obs2_buf, self.act_buf, self.rew_buf, self.done_buf]
        demo_data = {}
        for i in range(len(key)):
            demo_data[key[i]] = data[i]
        with open(output_file, 'wb') as fp:
            pickle.dump(demo_data, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, input_file):

        with open(input_file, 'rb') as fp:
            demo_data = pickle.load(fp)
        self.obs2_buf = np.array(demo_data['obs2_buf'])
        self.obs_buf = np.array(demo_data['obs_buf'])
        self.act_buf = np.array(demo_data['act_buf'])
        self.rew_buf = np.array(demo_data['rew_buf'])
        self.done_buf = np.array(demo_data['done_buf'])
        self.size = len(demo_data['done_buf'])


def merge_batch(batch1, batch2):
    keys = ['obs', 'act', 'rew', 'obs2', 'done']
    batch = {}
    for k in keys:
        batch[k] = torch.cat([batch1[k], batch2[k]], 0)
    return batch
