import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from memory import MemoryBuffer
from rnd import RND
from smirl import SMiRL
from model import ActorNN, CriticNN
from utils import *

from tqdm import tqdm
import os

update_proportion = 0.25

def add_smirl(s, smirl):
    aug_s_ = []
    for i, _s in enumerate(s):
        smirl[i].add(_s[8:408])
        aug_s = np.hstack([_s, smirl[i].get_params(), smirl[i].size - 1])
        aug_s_.append(aug_s)
    return np.stack(aug_s_)

class PPO:
    '''
    Multi-agent PPO with some possible RND or SMiRL option
    '''
    def __init__(self, model, lr, epsilon, lamda,
                 episode_num, episode_len,
                 norm_len, batch_size,
                 input_size, output_size,
                 gamma_ext, gamma_int,
                 v_coef, ent_coef,
                 ext_coef, int_coef,
                 epochs, workers, device,
                 use_rnd=False, rnd=None, use_smirl=False, smirl_arg=None,
                 group_actions=False):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.clip_ratio = epsilon
        self.lamda = lamda
        self.episode_num = episode_num
        self.episode_len = episode_len
        self.norm_len = norm_len
        self.batch_size = batch_size

        self.input_size = input_size
        self.output_size = output_size

        self.gamma_ext = gamma_ext
        self.gamma_int = gamma_int
        self.v_coef = v_coef
        self.ent_coef = ent_coef
        self.ext_coef = ext_coef
        self.int_coef = int_coef
        self.epochs = epochs

        self.use_rnd = use_rnd
        self.use_smirl = use_smirl

        self.workers = workers
        self.group_actions = group_actions
        self.device = device

        if use_rnd:
            self.rnd = rnd
            self.rnd_optimizer = optim.Adam(self.rnd.predictor.parameters(), lr=lr)

        if use_smirl:
            self.smirl_nn = [SMiRL(smirl_arg) for _ in range(workers)]
    
    def collect_state_statistics(self, envs):
        print('Start normalization')

        all_obs = []
        self.obs_rms = RunningMeanStd(shape=(self.input_size,))                     # add obs shape here
        self.reward_rms = RunningMeanStd()
        init = envs.get_recent()
        if self.use_smirl:
            init = add_smirl(init, self.smirl_nn)
        all_obs.append(init)
        self.ret = RewardForwardFilter(self.gamma_int)
        for _ in range(self.episode_len * self.norm_len):
            actions = np.random.randint(low=0, high=self.output_size, size=(envs.nenvs,))
            ns, _, done, _, _ = envs.step(actions)
            self.t += 1
            if self.use_smirl:
                ns = add_smirl(ns, self.smirl_nn)
                for i, _done in enumerate(done):
                    if _done:
                        self.smirl_nn[i].reset()
            all_obs.append(ns)
            if len(all_obs) % (self.episode_len * envs.nenvs) == 0:
                obs_ = np.asarray(all_obs).astype(np.float32).reshape((-1, self.input_size))
                self.obs_rms.update(obs_)
                all_obs.clear()
        
        print('Normalization done.')
    
    def step(self, envs):
        tot_s, tot_re, tot_done, tot_ns, tot_a = [], [], [], [], []
        tot_ri, tot_ve, tot_vi, tot_prob, tot_smirl_r = [], [], [], [], []
        actionss = [[]] * self.workers
        for _ in tqdm(range(self.episode_len),desc=f'Rollout',mininterval=0.5):
            s = envs.get_recent()
            if self.use_smirl:
                s = add_smirl(s, self.smirl_nn)
            a, ve, vi, _, _, result = self.model(s)
            if self.group_actions:
                for i, actions in enumerate(actionss):
                    if len(actions) == 0:
                        _, actions = self.model.best_state(*envs.get_all_next_hd(i))
                    a[i] = actions.pop()
            ns, re, done, _, _ = envs.step(a)
            if self.group_actions:
                for i, _done in enumerate(done):
                    if _done:
                        actionss[i].clear()

            if self.use_smirl:
                smirl_r = np.array([np.clip(smirl.logprob(ns[:,8:408]) / 1000., -5., 5.) for smirl in self.smirl_nn])
                re += smirl_r
                ns = add_smirl(ns, self.smirl_nn)
                tot_smirl_r.append(smirl_r)
                for i, _done in enumerate(done):
                    if _done:
                        self.smirl_nn[i].reset()

            if self.use_rnd:
                ri_rnd = self.rnd.eval_int((ns - self.obs_rms.mean / np.sqrt(self.obs_rms.var)))
                tot_ri.append(ri_rnd.data.cpu().numpy())

            tot_ns.append(ns)
            tot_s.append(s)
            tot_re.append(re)
            tot_done.append(done)
            tot_a.append(a.data.cpu().numpy().squeeze())
            tot_ve.append(ve.data.cpu().numpy().squeeze())
            tot_vi.append(vi.data.cpu().numpy().squeeze())
            tot_prob.append(result.detach().cpu())
        
        _, ve, vi, _, _, _ = self.model(s)
        tot_ve.append(ve.data.cpu().numpy().squeeze())
        tot_vi.append(vi.data.cpu().numpy().squeeze())

        tot_s = np.stack(tot_s).transpose([1,0,2]).reshape([-1, self.input_size])
        tot_ns = np.stack(tot_ns).transpose([1,0,2]).reshape([-1, self.input_size])
        tot_a = np.stack(tot_a).transpose().reshape([-1])
        tot_done = np.stack(tot_done).transpose()
        
        tot_re = np.stack(tot_re).transpose()
        tot_ve = np.stack(tot_ve).transpose()
        tot_vi = np.stack(tot_vi).transpose()
        if self.use_rnd:
            tot_ri = np.stack(tot_ri).transpose()

        tot_prob = np.stack(tot_prob).transpose([1,0,2]).reshape([-1, self.output_size])

        if self.use_rnd:
            # input_size = 457, output_size = 8
            rets = [self.ret.update(r) for r in tot_ri.T]
            mean, stdvar, count = np.mean(rets), np.std(rets), len(rets)
            self.reward_rms.update_from_moments(mean, stdvar ** 2, count)
            tot_ri /= np.sqrt(self.reward_rms.var)

            self.obs_rms.update(tot_ns)

            ns = (ns - self.obs_rms.mean / np.sqrt(self.obs_rms.var))
            ret_ext, adv_ext = eval_ret_adv(tot_re, tot_done, tot_ve, self.gamma_ext,
                                            self.episode_len, self.lamda, envs.nenvs)
            ret_int, adv_int = eval_ret_adv(tot_ri, np.zeros_like(tot_ri), tot_vi, self.gamma_int,
                                            self.episode_len, self.lamda, envs.nenvs)
            adv = torch.Tensor(self.int_coef * adv_int + self.ext_coef * adv_ext).unsqueeze(-1)
            ret_ext = torch.Tensor(ret_ext).unsqueeze(-1)
            ret_int = torch.Tensor(ret_int).unsqueeze(-1)
            s = torch.Tensor(tot_s)
            a = torch.Tensor(tot_a)
            ns = torch.Tensor(tot_ns)
            logprob = Categorical(torch.FloatTensor(tot_prob)).log_prob(a).unsqueeze(-1)
            a = a.unsqueeze(-1)

            result = self.update(s, a, ret_ext, ret_int, ns, adv, logprob)
        else:
            ret, adv = eval_ret_adv(tot_re, tot_done, tot_ve, self.gamma_ext, self.episode_len, self.lamda, envs.nenvs)
            adv = torch.Tensor(adv).unsqueeze(-1)
            ret = torch.Tensor(ret).unsqueeze(-1)
            s = torch.Tensor(tot_s)
            a = torch.Tensor(tot_a)
            logprob = Categorical(torch.FloatTensor(tot_prob)).log_prob(a).unsqueeze(-1)
            a = a.unsqueeze(-1)

            result = self.update_vanilla(s, a, ret, adv, logprob)
        
        if self.use_smirl:
            result['max_smirl_r'] = np.max(tot_smirl_r)
            result['min_smirl_r'] = np.min(tot_smirl_r)
            result['avg_smirl_r'] = np.mean(tot_smirl_r)
        mean_len = envs.nenvs * self.episode_len / (np.count_nonzero(tot_done) + envs.nenvs)
        if self.episode_len - mean_len < 10:
            envs.reset()
        result['episode_avg_len'] = mean_len
        return result

    def train(self, envs, save_dir, save_freq, writer):
        self.model.train()
        if self.use_rnd:
            self.rnd.train()

        self.t = 0
        if self.use_rnd:
            self.collect_state_statistics(envs)
            self.t -= self.norm_len * self.episode_len
        while True:
            result = self.step(envs)
            writer.add_scalars('train', result, global_step=self.workers * self.episode_len)

            if (self.t+1) % save_freq == 0:
                print(f'Episode {self.t+1} : {result}')
                self.save(save_dir, envs.name)
                if self.use_rnd:
                    self.rnd.save(save_dir, envs.name)

            if self.t > self.episode_num:
                break
            
            self.t += 1

        self.model.eval()
        if self.use_rnd:
            self.rnd.eval()

        return writer
    
    def update_vanilla(self, s, a, ret, adv, logprob):
        torch.autograd.set_detect_anomaly(True)
        dataset = TensorDataset(s, a, ret, adv, logprob)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for epoch in tqdm(range(self.epochs), desc=f'Update {self.t+1}',mininterval=0.5):
            actor_losses, critic_losses, entropy_bonuses = [], [], []
            for _s, _a, _ret, _adv, _log_prob in loader:
                action, value, _, log_prob, entropy, dist = self.model(_s)
                log_prob = Categorical(dist).log_prob(_a)
                ratio = (log_prob - _log_prob).exp()
                surr1 = _adv * ratio
                surr2 = _adv * torch.clamp(ratio, 1.0-self.clip_ratio, 1.0+self.clip_ratio)

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(value, _ret)
                entropy_bonus = -entropy.mean()

                self.optimizer.zero_grad()
                loss = actor_loss + self.v_coef * critic_loss + self.ent_coef * entropy_bonus
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                
                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropy_bonuses.append(-entropy_bonus.item())

            #print(f"[Epoch {epoch+1}] loss : {loss.item():.6f}")
        
        result = {'loss' : loss.item(),
                  'actor_loss' : np.mean(actor_losses),
                  'critic_loss' : np.mean(critic_losses),
                  'entropy_bonus' : np.mean(entropy_bonuses),
                  'min_return' : torch.min(ret).item(),
                  'max_return' : torch.max(ret).item(),
                  'avg_return' : torch.mean(ret).item()}

        return result

    def update(self, s, a, ret_ext, ret_int, ns, adv, logprob):
        torch.autograd.set_detect_anomaly(True)
        # print(s.shape, a.shape, ret_ext.shape, ret_int.shape, ns.shape, adv.shape, logprob.shape)
        dataset = TensorDataset(s, a, ret_ext, ret_int, ns, adv, logprob)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for epoch in tqdm(range(self.epochs),desc=f'Update {self.t+1}',mininterval=0.5):
            actor_losses, critic_losses, entropy_bonuses = [], [], []
            for _s, _a, _ret_ext, _ret_int, _ns, _adv, _log_prob in loader:
                predict_ns_feature, target_ns_feature = self.rnd(_ns)
                forward_loss = nn.MSELoss(reduction='none')(predict_ns_feature, target_ns_feature.detach()).mean(-1)
                mask = torch.rand(len(forward_loss))
                mask = (mask < update_proportion).type(torch.FloatTensor)
                forward_loss = (forward_loss * mask).sum() / torch.max(mask.sum(), torch.Tensor([1]))

                action, value_ext, value_int, log_prob, entropy, dist = self.model(_s)

                log_prob = Categorical(dist).log_prob(_a)
                ratio = (log_prob - _log_prob).exp()
                surr1 = _adv * ratio
                surr2 = _adv * torch.clamp(ratio, 1.0-self.clip_ratio, 1.0+self.clip_ratio)
                
                critic_ext_loss = F.mse_loss(value_ext, _ret_ext)
                critic_int_loss = F.mse_loss(value_int, _ret_int)

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = critic_ext_loss + critic_int_loss
                entropy_bonus = -entropy.mean()

                self.optimizer.zero_grad()
                self.rnd_optimizer.zero_grad()
                loss = actor_loss + self.v_coef * critic_loss + self.ent_coef * entropy_bonus + forward_loss
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                nn.utils.clip_grad_norm_(self.rnd.parameters(), 0.5)
                self.rnd_optimizer.step()
                self.optimizer.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropy_bonuses.append(-entropy_bonus.item())

            #print(f"[Epoch {epoch+1}] loss : {loss.item():.6f}")
        
        result = {'loss' : loss.item(),
                  'actor_loss' : np.mean(actor_losses),
                  'critic_loss' : np.mean(critic_losses),
                  'entropy_bonus' : np.mean(entropy_bonuses),
                  'max_return_extrinsic' : torch.max(ret_ext).item(),
                  'min_return_extrinsic' : torch.min(ret_ext).item(),
                  'avg_return_extrinsic' : torch.mean(ret_ext).item(),
                  'max_return_intrinsic' : torch.max(ret_int).item(),
                  'min_return_intrinsic' : torch.min(ret_int).item(),
                  'avg_return_intrinsic' : torch.mean(ret_int).item(),
                  'min_return' : torch.min(ret_ext + ret_int).item(),
                  'max_return' : torch.max(ret_ext + ret_int).item(),
                  'avg_return' : torch.mean(ret_ext + ret_int).item()}
        
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
