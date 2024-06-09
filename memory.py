import random
from collections import deque
import numpy as np
import torch

class MemoryBuffer:
    def __init__(self, use_intrinsic_reward=False):
        self.buffer = []
        self.use_intrinsic_reward = use_intrinsic_reward
    
    def push(self, *args):
        if self.use_intrinsic_reward:
            self._push_int(*args)
        else:
            self._push_ext(*args)

    def _push_ext(self, s, a, r, ns, done):
        transition = (s, a, r, ns, done)
        self.buffer.append(transition)

    def _push_int(self, s, a, re, ri, ns, done):
        transition = (s, a, re, ri, ns, done)
        self.buffer.append(transition)

    def sample(self):
        s = torch.Tensor(np.array([i[0] for i in self.buffer]))
        a = torch.Tensor(np.array([[i[1]] for i in self.buffer]))
        re = torch.Tensor(np.array([[i[2]] for i in self.buffer]))
        if self.use_intrinsic_reward:
            ri = torch.Tensor(np.array([[i[3]] for i in self.buffer]))
            ns = torch.Tensor(np.array([i[4] for i in self.buffer]))
            done = torch.Tensor(np.array([[i[5]] for i in self.buffer]))
            return s, a, re, ri, ns, done
        ns = torch.Tensor(np.array([i[3] for i in self.buffer]))
        done = torch.Tensor(np.array([[i[4]] for i in self.buffer]))
        #goal = np.asarray([i[5] for i in batch])

        return s, a, re, ns, done

    @property
    def size(self):
        return len(self.buffer)
    
    def clear(self):
        self.buffer = []

    def set_use_int(self):
        self.use_intrinsic_reward = True
        self.clear()


class ReplayBuffer:
    def __init__(self, capacity, use_intrinsic_reward=False):
        self.capacity = capacity
        self.buffer = deque()
        self.use_intrinsic_reward = use_intrinsic_reward
        self.count = 0

    def push(self, *args):
        if self.use_intrinsic_reward:
            self._push_int(*args)
        else:
            self._push_ext(*args)

    def _push_ext(self, s, a, r, ns, done):
        transition = (s, a, r, ns, done)

        if self.count < self.capacity:
            self.buffer.append(transition)
            self.count = self.count + 1
        else:
            self.buffer.popleft()
            self.buffer.append(transition)

    def _push_int(self, s, a, re, ri, ns, done):
        transition = (s, a, re, ri, ns, done)

        if self.count < self.capacity:
            self.buffer.append(transition)
            self.count = self.count + 1
        else:
            self.buffer.popleft()
            self.buffer.append(transition)

    def sample(self, batch_size):
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)
        s = torch.Tensor(np.array([i[0] for i in batch]))
        a = torch.Tensor(np.array([[i[1]] for i in batch]))
        r = torch.Tensor(np.array([[i[2]] for i in batch]))
        ns = torch.Tensor(np.array([i[3] for i in batch]))
        done = torch.Tensor(np.array([[i[4]] for i in batch]))
        #goal = np.asarray([i[5] for i in batch])

        return s, a, r, ns, done

    def all(self):
        s = torch.Tensor(np.array([i[0] for i in self.buffer]))
        a = torch.Tensor(np.array([i[1] for i in self.buffer]))
        r = torch.Tensor(np.array([[i[2]] for i in self.buffer]))
        ns = torch.Tensor(np.array([i[3] for i in self.buffer]))
        done = torch.Tensor(np.array([[i[4]] for i in self.buffer]))
        #goal = np.asarray([i[5] for i in batch])

        return s, a, r, ns, done
    
    @property
    def size(self):
        return self.count
    
    def clear(self):
        self.buffer = deque()
        self.count = 0

    def set_use_int(self):
        self.use_intrinsic_reward = True
        self.clear()


class PriorityReplayBuffer:
    def __init__(self):
        pass