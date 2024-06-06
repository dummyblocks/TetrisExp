import torch
import numpy as np

REG = 255.

class SMiRL:
    def __init__(self, dim):
        self.buffer = np.zeros(dim)
        self.dim = dim
        self.size = 1

    def add(self, s):
        # Tetris environment has observation space of..
        # field(200) / field_view
        # mino_pos / mino_rot / mino / hold / preview / status
        # we only need field for minimizing surprise

        self.buffer += s[:200] / 3.
        self.size += 1

    def logprob(self, s):
        # look for field_view for comparison.
        s = s.copy()[200:400] / 3.
        theta = self.get_params()
        theta = np.clip(theta, 1e-5, 1 - 1e-5)
        log_probs = s * np.log(theta) + (1-s) * np.log(1-theta)
        return np.sum(log_probs)
    
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