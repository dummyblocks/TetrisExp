import torch
import numpy as np

REG = 255.

class SMiRL:
    def __init__(self, dim):
        self.buffer = np.zeros(dim)
        self.dim = dim
        self.size = 1

    def add(self, img):
        # Tetris environment has observation space of..
        # img / mino_pos / mino_rot / mino / hold / preview / status
        # we only need img for minimizing surprise
        # Aware that img must be flattened.
        self.buffer += img
        self.size += 1

    def logprob(self, img):
        # look for field_view for comparison.
        theta = self.get_params()
        theta = np.clip(theta, 1e-5, 1 - 1e-5)
        log_probs = img * np.log(theta) + (1-img) * np.log(1-theta)
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