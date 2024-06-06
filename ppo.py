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

def add_smirl(s, smirl, t):
    aug_s_ = []
    for i, _s in enumerate(s):
        smirl[i].add(_s)
        aug_s = np.hstack([_s, smirl[i].get_params(), t])
        aug_s_.append(aug_s)
    return np.stack(aug_s_)


class PPO:
    '''
    Multi-agent PPO with some possible RND or SMiRL option
    '''
    def __init__(self, actor: ActorNN, critic: CriticNN, actor_lr, critic_lr, epsilon,
                 episode_num, episode_len, norm_len, batch_size, lamda, #memory_size, lamda,
                 input_size, output_size,
                 gamma_ext, gamma_int,
                 v_coef, ent_coef,
                 ext_coef, int_coef,
                 epochs, workers,
                 use_rnd=False, rnd_arg=None, use_smirl=False, smirl_arg=None):
        self.actor = actor
        self.critic = critic
        self.actor_optimizer = optim.Adam(self.actor.parameters(), actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), critic_lr)
        self.clip_ratio = epsilon
        #self.memory = MemoryBuffer()
        self.episode_num = episode_num
        self.episode_len = episode_len
        self.norm_len = norm_len
        self.batch_size = batch_size

        self.input_size = input_size
        self.output_size = output_size

        self.lamda = lamda
        self.gamma_ext = gamma_ext
        self.gamma_int = gamma_int
        self.v_coef = v_coef
        self.ent_coef = ent_coef
        self.ext_coef = ext_coef
        self.int_coef = int_coef
        self.epochs = epochs

        self.use_rnd = use_rnd
        self.use_smirl = use_smirl

        if use_rnd:
            self.rnd_network = RND(*rnd_arg)
            self.rnd_optimizer = optim.Adam(self.rnd_network.predictor.parameters(), critic_lr)
            #self.memory.set_use_int()

        if use_smirl:
            self.smirl_nn = [SMiRL(smirl_arg) for _ in range(workers)]
    
    def collect_state_statistics(self, envs):
        print('Start normalization')

        all_obs = []
        self.obs_rms = RunningMeanStd(shape=(self.input_size,))                     # add obs shape here
        self.reward_rms = RunningMeanStd()
        init = envs.get_recent()
        if self.use_smirl:
            init = add_smirl(init, self.smirl_nn, self.t)
        all_obs.append(init)
        self.ret = RewardForwardFilter(self.gamma_int)
        for _ in range(self.episode_len * self.norm_len):
            actions = np.random.randint(low=0, high=self.output_size, size=(envs.nenvs,))
            ns, _, done, _, _ = envs.step(actions)
            self.t += 1
            if self.use_smirl:
                ns = add_smirl(ns, self.smirl_nn, self.t)
            all_obs.append(ns)
            if len(all_obs) % (self.episode_len * envs.nenvs) == 0:
                obs_ = np.asarray(all_obs).astype(np.float32).reshape((-1, self.input_size))
                self.obs_rms.update(obs_)
                all_obs.clear()
        
        print('Normalization done.')
    
    def step(self, envs):
        tot_s, tot_re, tot_done, tot_ns, tot_a = [], [], [], [], []
        tot_ri, tot_ve, tot_vi, tot_prob, tot_smirl_r = [], [], [], [], []
        for _ in range(self.episode_len):
            s = envs.get_recent()
            if self.use_smirl:
                s = add_smirl(s, self.smirl_nn, self.t)
            a, _, _, result = self.actor(s)
            ve, vi = self.critic(s)
            ns, re, done, _, _ = envs.step(a)

            if self.use_smirl:
                aug_ns_, smirl_r = [], []
                for i, (_s, _ns) in enumerate(zip(s,ns)):
                    ri_smirl = np.clip(self.smirl_nn[i].logprob(_ns) / 1000., -1., 1.)
                    re[i] += ri_smirl
                    self.smirl_nn[i].add(_ns)
                    aug_ns = np.hstack([_ns, self.smirl_nn[i].get_params(), _s[-1]+1])
                    aug_ns_.append(aug_ns)
                    smirl_r.append(ri_smirl)
                ns = np.stack(aug_ns_)
                #self.memory.push(s, a, re, ns, done)
                tot_smirl_r.append(np.stack(smirl_r))

            if self.use_rnd:
                ri_rnd = self.rnd_network.eval_int(((ns - self.obs_rms.mean) / np.sqrt(self.obs_rms.var)).clip(-5, 5))
                tot_ri.append(ri_rnd.data.cpu().numpy())

            tot_ns.append(ns)
            tot_s.append(s)
            tot_re.append(re)
            tot_done.append(done)
            tot_a.append(a.data.cpu().numpy().squeeze())
            tot_ve.append(ve.data.cpu().numpy().squeeze())
            if self.use_rnd:
                tot_vi.append(vi.data.cpu().numpy().squeeze())
            tot_prob.append(result.detach().cpu())

            # if not self.use_rnd and not self.use_smirl:
                #self.memory.push(s, a, re, ns, done)
        
        ve, vi = self.critic(s)
        tot_ve.append(ve.data.cpu().numpy().squeeze())
        if self.use_rnd:
            tot_vi.append(vi.data.cpu().numpy().squeeze())

        tot_s = np.stack(tot_s).transpose([1,0,2]).reshape([-1, self.input_size])
        tot_ns = np.stack(tot_ns).transpose([1,0,2]).reshape([-1, self.input_size])
        tot_a = np.stack(tot_a).transpose().reshape([-1])
        tot_done = np.stack(tot_done).transpose()
        
        tot_re = np.stack(tot_re).transpose()
        tot_ve = np.stack(tot_ve).transpose()
        if self.use_rnd:
            tot_ri = np.stack(tot_ri).transpose()
            tot_vi = np.stack(tot_vi).transpose()

        tot_prob = np.stack(tot_prob).transpose([1,0,2]).reshape([-1, self.output_size])

        if self.use_rnd:
            # input_size = 457, output_size = 8
            rets = [self.ret.update(r) for r in tot_ri.T]
            mean, stdvar, count = np.mean(rets), np.std(rets), len(rets)
            self.reward_rms.update_from_moments(mean, stdvar ** 2, count)
            tot_ri /= np.sqrt(self.reward_rms.var)

            self.obs_rms.update(tot_ns)

            ns = ((tot_ns - self.obs_rms.mean) / np.sqrt(self.obs_rms.var)).clip(-5, 5)
            ret_ext, adv_ext = eval_ret_adv(tot_re, tot_done, tot_ve, self.gamma_ext,
                                            self.episode_len, self.lamda, envs.nenvs)
            ret_int, adv_int = eval_ret_adv(tot_ri, np.zeros_like(tot_ri), tot_vi, self.gamma_int,
                                            self.episode_len, self.lamda, envs.nenvs)
            adv = torch.Tensor(self.int_coef * adv_int + self.ext_coef * adv_ext).unsqueeze(-1)
            ret_ext = torch.Tensor(ret_ext).unsqueeze(-1)
            ret_int = torch.Tensor(ret_int).unsqueeze(-1)
            s = torch.Tensor(tot_s)
            a = torch.Tensor(tot_a)
            ns = torch.Tensor(ns)
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
        result['episode_avg_len'] = envs.nenvs * self.episode_len / np.sum(tot_done)
        return result

    def train(self, envs, save_dir, save_freq, writer):
        self.t = 0
        if self.use_rnd:
            self.collect_state_statistics(envs)
            self.t -= self.norm_len * self.episode_len
        while True:
            # self.memory.clear()
            if self.use_smirl:
                for smirl in self.smirl_nn:
                    smirl.reset()

            result = self.step(envs)
            writer.add_scalars('train', result)

            if (self.t+1) % save_freq == 0:
                print(f'Episode {self.t+1} : {result}')
                self.save(save_dir, envs.name)
                if self.use_rnd:
                    self.rnd_network.save(save_dir, envs.name)
                if self.use_smirl:
                    for smirl in self.smirl_nn:
                        smirl.save(save_dir, envs.name)

            if self.t > self.episode_num:
                break
            
            self.t += 1

        return writer
    
    def update_vanilla(self, s, a, ret, adv, logprob):
        dataset = TensorDataset(s, a, ret, adv, logprob)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for epoch in tqdm(range(self.epochs), desc=f'Update {self.t+1}',mininterval=0.5):
            actor_losses, critic_losses, entropy_bonuses = [], [], []
            for _s, _a, _ret, _adv, _log_prob in loader:
                value, _ = self.critic(_s)
                action, log_prob, entropy, dist = self.actor(_s)
                log_prob = Categorical(dist).log_prob(_a)
                ratio = (log_prob - _log_prob).exp()
                surr1 = _adv * ratio
                surr2 = _adv * torch.clamp(ratio, 1.0-self.clip_ratio, 1.0+self.clip_ratio)

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(value, _ret)
                entropy_bonus = -entropy.mean()

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()

                loss = actor_loss + self.v_coef * critic_loss + self.ent_coef * entropy_bonus
                loss.backward()

                self.critic_optimizer.step()
                self.actor_optimizer.step()

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
        dataset = TensorDataset(s, a, ret_ext, ret_int, ns, adv, logprob)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for epoch in tqdm(range(self.epochs),desc=f'Update {self.t+1}',mininterval=0.5):
            actor_losses, critic_losses, entropy_bonuses = [], [], []
            for _s, _a, _ret_ext, _ret_int, _ns, _adv, _log_prob in loader:
                predict_ns_feature, target_ns_feature = self.rnd_network(_ns)
                forward_loss = nn.MSELoss(reduction='none')(predict_ns_feature, target_ns_feature.detach()).mean(-1)
                mask = torch.rand(len(forward_loss))
                mask = (mask < update_proportion).type(torch.FloatTensor)
                forward_loss = (forward_loss * mask).sum() / torch.max(mask.sum(), torch.Tensor([1]))

                value_ext, value_int = self.critic(_s)
                critic_ext_loss = F.mse_loss(value_ext, _ret_ext)
                critic_int_loss = F.mse_loss(value_int, _ret_int)

                action, log_prob, entropy, dist = self.actor(_s)
                log_prob = Categorical(dist).log_prob(_a)
                ratio = (log_prob - _log_prob).exp()
                surr1 = _adv * ratio
                surr2 = _adv * torch.clamp(ratio, 1.0-self.clip_ratio, 1.0+self.clip_ratio)

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = critic_ext_loss + critic_int_loss
                entropy_bonus = -entropy.mean()

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                self.rnd_optimizer.zero_grad()

                loss = actor_loss + self.v_coef * critic_loss + self.ent_coef * entropy_bonus + forward_loss
                loss.backward()

                self.rnd_optimizer.step()
                self.critic_optimizer.step()
                self.actor_optimizer.step()

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
        actor_path = os.path.join(save_dir, env_name + '.modela')
        critic_path = os.path.join(save_dir, env_name + '.modela')
        if use_cuda:
            self.actor.load_state_dict(torch.load(actor_path))
            self.critic.load_state_dict(torch.load(critic_path))
        else:
            self.actor.load_state_dict(torch.load(actor_path, map_location='cpu'))
            self.critic.load_state_dict(torch.load(critic_path, map_location='cpu'))

    def save(self, save_dir, env_name):
        actor_path = os.path.join(save_dir, env_name + '.modela')
        critic_path = os.path.join(save_dir, env_name + '.modelc')
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

