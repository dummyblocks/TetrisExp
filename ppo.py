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
import os

update_proportion = 0.25

class PPO:
    '''
    Multi-agent PPO with some possible RND or SMiRL
    '''

    t = 0


    def __init__(self, actor: ActorNN, critic: CriticNN, actor_lr, critic_lr, epsilon,
                 episode_num, episode_len, norm_len, batch_size, lamda, #memory_size, lamda,
                 gamma_ext, gamma_int, v_coef, ent_coef, ext_coef, int_coef, epochs,
                 use_rnd=False, rnd_arg=None, use_smirl=False, smirl_arg=None):
        self.actor = actor
        self.critic = critic
        self.actor_optimizer = optim.Adam(self.actor.parameters(), actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), critic_lr)
        self.clip_ratio = epsilon
        self.memory = MemoryBuffer()
        self.episode_num = episode_num
        self.episode_len = episode_len
        self.norm_len = norm_len
        self.batch_size = batch_size

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
            self.memory.set_use_int()

        if use_smirl:
            self.smirl_nn = SMiRL(smirl_arg)

    @torch.no_grad()
    def act(self, state, training=True):
        self.policy.train(training)

        action, log_prob, entropy = self.actor(state)

        return action
    
    def collect_state_statistics(self, envs, init):
        print('Start normalization')

        action_space = envs.action_space()
        observ_space = envs.observation_space()
        all_obs = []
        self.obs_rms = RunningMeanStd()                     # add obs shape here
        self.reward_rms = RunningMeanStd()
        self.ret = RewardForwardFilter(self.gamma_int)
        all_obs.append(init)
        for _ in range(self.episode_len * self.norm_len):
            actions = np.random.randint(low=0, high=action_space.n, size=(envs.nenvs,))
            ns, _, done, _, _ = envs.step(actions)
            all_obs.append(ns)
            for i, term in enumerate(done):
                if term:
                    envs.reset_idx(i)
            if len(all_obs) % (self.episode_len * envs.nenvs) == 0:
                obs_ = np.asarray(all_obs).astype(np.float32).reshape((-1, *observ_space.shape))
                self.obs_rms.update(obs_)
                all_obs.clear()
        
        print('Normalization done.')
    
    def step(self, envs, s):
        tot_s, tot_re, tot_done, tot_ns, tot_a = [], [], [], [], []
        tot_ri, tot_ve, tot_vi, tot_prob, tot_smirl_r = [], [], [], [], []
        for t in range(self.episode_len):
            a, _, _, result = self.actor(s)
            ve, vi = self.critic(s)
            ns, re, done, _, _ = envs.step(a)

            if self.use_smirl:
                aug_ns_ = []
                smirl_r = []
                for i, (_s, _ns) in enumerate(zip(s, ns)):
                    ri_smirl = self.smirl_nn.logprob(_ns)
                    re[i] += ri_smirl
                    aug_s = np.hstack([_s, self.smirl_nn.get_params(), t])
                    self.smirl_nn.add(_ns)
                    aug_ns = np.hstack([_ns, self.smirl_nn.get_params(), t+1])
                    aug_ns_.append(aug_ns)
                    self.memory.push(_s, a[i], re[i], aug_ns, done[i])
                    smirl_r.append(ri_smirl)
                ns = np.stack(aug_ns_)
                tot_smirl_r.append(np.stack(smirl_r))

            if self.use_rnd:
                ri_rnd = self.rnd_network.eval_int(((ns - self.obs_rms.mean) / np.sqrt(self.obs_rms.var)).clip(-5, 5))

                tot_ns.append(ns)
                tot_ri.append(ri_rnd.data.cpu().numpy())
                tot_s.append(s)
                tot_re.append(re)
                tot_done.append(done)
                tot_a.append(a.data.cpu().numpy().squeeze())
                tot_ve.append(ve.data.cpu().numpy().squeeze())
                tot_vi.append(vi.data.cpu().numpy().squeeze())
                tot_prob.append(result.detach().cpu())

            if not self.use_rnd and not self.use_smirl:
                for _s, _a, _re, _ns, _done in zip(s, a, re, ns, done):
                    self.memory.push(_s, _a, _re, _ns, _done)

            s = ns
            if t < self.episode_len - 1:
                for i, term in enumerate(done):
                    if term:
                        # observation_size = 457
                        s[i, :457] = envs.reset_idx(i)

        if self.use_smirl:
            print(tot_smirl_r[0])

        if self.use_rnd:
            ve, vi = self.critic(s)
            tot_ve.append(ve.data.cpu().numpy().squeeze())
            tot_vi.append(vi.data.cpu().numpy().squeeze())
            
            # input_size = 457, output_size = 8
            tot_s = np.stack(tot_s).transpose([1,0,2]).reshape([-1, 457])
            tot_ns = np.stack(tot_ns).transpose([1,0,2]).reshape([-1, 457])
            tot_a = np.stack(tot_a).transpose().reshape([-1])
            tot_done = np.stack(tot_done).transpose()
            tot_ve = np.stack(tot_ve).transpose()
            tot_vi = np.stack(tot_vi).transpose()
            tot_re = np.stack(tot_re).transpose()
            tot_ri = np.stack(tot_ri).transpose()
            tot_prob = np.stack(tot_prob).transpose([1,0,2]).reshape([-1, 8])
    
            rets = [self.ret.update(r) for r in tot_ri.T]
            mean, stdvar, count = np.mean(rets), np.std(rets), len(rets)
            self.reward_rms.update_from_moments(mean, stdvar ** 2, count)
            tot_ri /= np.sqrt(self.reward_rms.var)

            self.obs_rms.update(tot_ns)

            return self.update(tot_s, tot_a, tot_ns, tot_re, tot_ri,
                               tot_ve, tot_vi, tot_done, tot_prob, envs.nenvs), s
        else:
            return self.update_vanilla(), s

    def train(self, envs, save_dir, save_freq, writer):
        state = envs.reset()
        if self.use_smirl:
            state = np.stack([np.hstack([s, self.smirl_nn.get_params(), 0]) for s in state])
        if self.use_rnd:
            self.collect_state_statistics(envs, state)
        t = 1
        while True:
            self.memory.clear()

            result, state = self.step(envs, state)

            if t % save_freq == 0:
                print(f'Episode {t} : {result}')
                self.save(save_dir, envs.name)
                if self.use_rnd:
                    self.rnd_network.save(save_dir, envs.name)
                if self.use_smirl:
                    self.smirl_nn.save(save_dir, envs.name)

            if t > self.episode_num:
                break
            
            t += 1
    
    def update_vanilla(self):
        s, a, r, ns, done = self.memory.sample()

        with torch.no_grad():
            value_ns, _ = self.critic(ns)
            value_s, _ = self.critic(s)
            delta = r + done * self.gamma_ext * value_ns - value_s
            adv = torch.clone(delta)
            ret = torch.clone(r)
            for t in range(len(r)-2, -1, -1):
                adv[t] += (1 - done[t]) * self.gamma_ext * self.lamda * adv[t+1]
                ret[t] += (1 - done[t]) * self.gamma_ext * ret[t+1]
            
            action, log_prob, entropy, dist = self.actor(s)

        dataset = TensorDataset(s, a, ret, ns, adv, log_prob)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for epoch in range(self.epochs):
            actor_losses, critic_losses, entropy_bonuses = [], [], []
            for _s, _a, _ret, _ns, _adv, _log_prob in loader:
                value, _ = self.critic(_s)
                action, log_prob, entropy, dist = self.actor(_s)
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

            print(f"[Epoch {epoch+1}] loss : {loss.item():.6f}")
        
        result = {'actor_loss' : np.mean(actor_losses),
                  'critic_loss' : np.mean(critic_losses),
                  'entropy_bonus' : np.mean(entropy_bonuses),
                  'min_return' : torch.min(ret).item(),
                  'max_return' : torch.max(ret).item(),
                  'avg_return' : torch.mean(ret).item()} 

        return result

    def update(self, s, a, ns, re, ri, ve, vi, done, prob, workers):
        ns = ((ns - self.obs_rms.mean) / np.sqrt(self.obs_rms.var)).clip(-5, 5)
        target_ext, adv_ext = eval_ret_adv(re, done, ve, self.gamma_ext,
                                           self.episode_len, self.lamda, workers)
        target_int, adv_int = eval_ret_adv(ri, np.zeros_like(ri), vi, self.gamma_int,
                                           self.episode_len, self.lamda, workers)
        adv = torch.Tensor(self.int_coef * adv_int + self.ext_coef * adv_ext).unsqueeze(-1)
        target_ext = torch.Tensor(target_ext).unsqueeze(-1)
        target_int = torch.Tensor(target_int).unsqueeze(-1)
        s = torch.Tensor(s)
        a = torch.Tensor(a)
        ns = torch.Tensor(ns)
        logprob = Categorical(torch.FloatTensor(prob)).log_prob(a).unsqueeze(-1)
        a = a.unsqueeze(-1)

        dataset = TensorDataset(s, a, target_ext, target_int, ns, adv, logprob)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for epoch in range(self.epochs):
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

                action, log_prob_, entropy, result = self.actor(_s)
                log_prob = Categorical(result).log_prob(_a)
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

            print(f"[Epoch {epoch+1}] loss : {loss.item():.6f}")
        
        result = {'actor_loss' : np.mean(actor_losses),
                  'critic_loss' : np.mean(critic_losses),
                  'entropy_bonus' : np.mean(entropy_bonuses),
                  'max_return_extrinsic' : torch.max(target_ext).item(),
                  'max_return_intrinsic' : torch.max(target_int).item(),
                  'min_return' : torch.min(target_ext + target_int).item(),
                  'max_return' : torch.max(target_ext + target_int).item(),
                  'avg_return' : torch.mean(target_ext + target_int).item()}
        
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

