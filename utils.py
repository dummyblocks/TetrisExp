import numpy as np
import torch

use_gae = True

def eval_ret_adv(reward, done, value, gamma, num_step, lamda, num_worker):
    ret = np.zeros([num_worker, num_step])

    if use_gae:
        gae = np.zeros_like([num_worker, ])
        for t in range(num_step - 1, -1, -1):
            delta = reward[:, t] + gamma * value[:, t + 1] * (1 - done[:, t]) - value[:, t]
            gae = delta + gamma * lamda * (1 - done[:, t]) * gae
            ret[:, t] = gae + value[:, t]
        adv = ret - value[:, :-1]
    else:
        running_add = value[:, -1]
        for t in range(num_step - 1, -1, -1):
            running_add = reward[:, t] + gamma * running_add * (1 - done[:, t])
            ret[:, t] = running_add
        adv = ret - value[:, :-1]

    return ret.reshape([-1]), adv.reshape([-1])

def parse_dict(state_dicts, use_smirl):
    result = []
    if type(state_dicts) == list and type(state_dicts[0]) == dict:
        img = np.stack([state['image'] for state in state_dicts])
        mino_pos = np.stack([state['mino_pos'] for state in state_dicts])
        mino_rot = np.stack([np.eye(4)[state['mino_rot']] for state in state_dicts])
        mino = np.stack([np.eye(7)[state['mino']] for state in state_dicts])
        hold = np.stack([np.eye(7)[state['hold']] for state in state_dicts])
        preview = np.stack([np.eye(7)[state['preview']].flatten() for state in state_dicts])
        status = np.stack([state['status'] for state in state_dicts])
        result.extend([img, mino_pos, mino_rot, mino, hold, preview, status])
        result.append(np.stack([state['smirl'] for state in state_dicts]) if use_smirl else None)
    else:
        # state_dicts is in shape of (~, 400 + 2 + 1 + 7 + 7 + 35 + 4)
        hold = torch.Tensor(state_dicts[:, :8])
        img = torch.Tensor(state_dicts[:, 8:408].reshape([-1, 1, 20, 20]))
        mino = torch.Tensor(state_dicts[:, 408:415])
        mino_pos = torch.Tensor(state_dicts[:, 415:417])
        mino_rot = torch.Tensor(state_dicts[:, 417:421])
        preview = torch.Tensor(state_dicts[:, 421:456])
        status = torch.Tensor(state_dicts[:, 456:460])
        result.extend([img, mino_pos, mino_rot, mino, hold, preview, status])
        result.append(torch.Tensor(state_dicts[:, 460:961]) if use_smirl else None)
    
    return tuple(result)
    

class RunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

class RewardForwardFilter(object):
    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma

    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems