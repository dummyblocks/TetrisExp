import torch
import numpy as np

class SMiRL:
    def __init__(self, dim):
        self.buffer = np.zeros(dim)
        self.dim = dim
        self.size = 1

    def add(self, s):
        # Tetris environment has observation space of..
        # field(200) / field_view
        # mino_pos / mino_rot / mino / hold / preview / status
        # we only need field_view for minimizing surprise

        self.buffer += s[200:400]
        self.size += 1

    def logprob(self, s):
        s = s.copy()[200:400]
        theta = self.get_params()
        theta = np.clip(theta, 1e-5, 1 - 1e-5)
        probs = s * theta + (1-s) * (1-theta)
        return np.sum(np.log(probs))
    
    def entropy(self):
        theta = self.get_params()
        theta = np.clip(theta, 1e-5, 1 - 1e-5)
        return np.sum(-theta * np.log(theta) - (1-theta) * np.log(1-theta))

    def get_params(self):
        theta = np.array(self.buffer) / self.size
        theta = np.clip(theta, 1e-5, 1 - 1e-5)
        return theta
    
    def reset(self):
        self.buffer = np.zeros(self.dim)
        self.size = 1

    def save(self, save_dir, env_name):
        import os
        smirl_path = os.path.join(save_dir, env_name + '.smirl')
        torch.save({'params':self.get_params()}, smirl_path)