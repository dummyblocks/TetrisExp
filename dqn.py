import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from smirl import SMiRL
from model import TetrisQ
from utils import *

from tqdm import tqdm
import random
import os

update_proportion = 0.25

def add_smirl(s, smirl):
    aug_s_ = []
    for i, _s in enumerate(s):
        smirl[i].add(_s[8:408])
        aug_s = np.hstack([_s, smirl[i].get_params(), smirl[i].size - 1])
        aug_s_.append(aug_s)
    return np.stack(aug_s_)

class TetrisDoubleDQN:
    '''
    Double DQN with some possible HER and PER & SMiRL.
    '''
    def __init__(self, model: TetrisQ, target: TetrisQ, lr, gamma, epsilon, epsilon_min, epsilon_stop, nstep,
                 batch_size, learning_start, learning_freq, memory, replay_size,
                 epochs, device, use_smirl=False, smirl_arg=None, group_actions=False):
        self.model = model
        self.target = target
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = (epsilon - epsilon_min) / epsilon_stop
        self.nstep = nstep
        self.batch_size = batch_size

        self.learning_start = learning_start
        self.learning_freq = learning_freq
        self.memory = memory

        self.gamma = gamma
        self.epochs = epochs
        self.replay_size = replay_size
        self.use_smirl = use_smirl

        self.group_actions = group_actions
        self.device = device

        if use_smirl:
            self.smirl = SMiRL(smirl_arg)

    def train(self, env, save_dir, save_freq, writer):
        self.model.train()
        self.target.train()

        ep_len = []
        ep_r = []
        ep_score = []

        s = env.reset()
        for t in range(self.nstep):
            if t < self.learning_start:
                a = np.random.randint(env.action_space.n)
            else:
                if random.random() > self.epsilon:
                    a = self.model(s).detach().data.max(1)[1][0]
                else:
                    a = np.random.randint(env.action_space.n)
            ns, r, done, _, info = env.step(a)
            self.memory.push(s, a, r, ns, done)

            if done:
                s = env.reset()
                ep_len.append(info['episode_len'])
                ep_r.append(info['episode_reward'])
                ep_score.append(info['score'])
            else:
                s = ns

            if t > self.learning_start and (t+1) % self.learning_freq == 0:
                result = self.update()
                result['episode_mean_len'] = np.mean(ep_len)
                result['episode_mean_reward'] = np.mean(ep_r)
                result['episode_mean_score'] = np.mean(ep_score)
                writer.add_scalars('train', result, global_step=t)

            if (t+1) % save_freq == 0:
                print(f'Episode {self.t+1} : {result}')
                self.save(save_dir, env.name)

        self.model.eval()
        self.target.eval()
    
    def update(self):
        if self.memory.size < self.replay_size:
            return
        
        s, a, r, ns, done = self.memory.sample(self.batch_size)
        pbar = tqdm(range(self.epochs),ncols=100,ascii='=',mininterval=0.5)
        for epoch in pbar:
            pbar.set_description(f'Epoch {epoch}/self.epochs')
            self.optimizer.zero_grad()
            value = self.model(s).gather(1, a).squeeze()
            value_next = self.model(ns).gather(1, self.target(ns).detach().gather(1, a).squeeze()).squeeze()
            loss = r + (1 - done) * self.gamma * value_next - value
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()

            self.num_update += 1

            if self.num_update % self.target_update_freq == 0:
                self.target.load_state_dict(self.model.state_dict())
        
        result = {'loss' : loss,}
        
        return result

    def load(self, save_dir, env_name, use_cuda):
        model_path = os.path.join(save_dir, env_name + '.model')
        if use_cuda:
            self.model.load_state_dict(torch.load(model_path))
        else:
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))

    def save(self, save_dir, env_name):
        model_path = os.path.join(save_dir, env_name + '.model')
        torch.save(self.model.state_dict(), model_path)
