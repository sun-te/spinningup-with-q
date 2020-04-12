import gym
from gym.utils import seeding
import numpy as np
import pickle
import torch

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


class DemoGymEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, demo_file):
        self.demo_file = demo_file

    def load_replay(self):
        return
