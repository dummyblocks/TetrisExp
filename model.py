import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from utils import parse_dict, parse_dict_simple
import random

class ActorNN(nn.Module):
    '''
    A linear discrete policy network
    '''
    def __init__(self, input_size, hidden_size, output_size, num_layers, activation, noisy):
        super().__init__()
        linear = NoisyLinear if noisy else nn.Linear
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size // 2),
            activation,
            nn.Linear(hidden_size // 2, hidden_size),
            activation,
        )
        for _ in range(num_layers - 3):
            self.model.append(nn.Sequential(
                linear(hidden_size, hidden_size),
                activation,
            ))
        self.model.append(linear(hidden_size, output_size))

    def forward(self, s):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, dtype=torch.float32)
            #s = s.squeeze(-1).unsqueeze(0)
        result = F.softmax(self.model(s), dim=1)
        action_dist = Categorical(result)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action).unsqueeze(1)
        entropy = action_dist.entropy().unsqueeze(1)

        return action.detach(), log_prob.detach(), entropy.detach(), result
    
    def evaluate(self, s, a):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, dtype=torch.float32)
            #s = s.squeeze(-1).unsqueeze(0)

        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a, dtype=torch.float32)
            #a = a.squeeze(-1).unsqueeze(0)

        action_dist = Categorical(F.softmax(self.model(s), dim=1))
        log_prob = action_dist.log_prob(a)
        entropy = action_dist.entropy()

        return log_prob.detach(), entropy.detach()


class CriticNN(nn.Module):
    '''
    A linear discrete value network
    '''
    def __init__(self, input_size, hidden_size, output_size, num_layers, activation, use_noisy=False, use_int=False):
        super().__init__()

        if use_noisy:
            linear = NoisyLinear
        else:
            linear = nn.Linear

        self.feature = nn.Sequential(
            nn.Linear(input_size, hidden_size // 2),
            activation,
            nn.Linear(hidden_size // 2, hidden_size),
            activation,
        )
        for _ in range(num_layers - 3):
            self.feature.append(nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                activation,
            ))
        self.use_int = use_int

        self.critic_ext = linear(hidden_size, output_size)

        if use_int:
            self.extra_layer = nn.Sequential(
                linear(hidden_size, hidden_size),
                activation
            )
            self.critic_int = linear(hidden_size, output_size)

    def forward(self, s):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, dtype=torch.float32)
            #s = s.squeeze(-1).unsqueeze(0)

        x = self.feature(s)
        if self.use_int:
            value_ext = self.critic_ext(self.extra_layer(x) + x)
            value_int = self.critic_int(self.extra_layer(x) + x)
            return value_ext, value_int
        else:
            value = self.critic_ext(x)
            return value, None
        
    def use_intrinsic_r(self):
        return self.use_int


class NoisyLinear(nn.Module):
    '''
    A simple Gaussian noisy linear net
    '''
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, input) -> torch.Tensor:
        if not isinstance(input, torch.Tensor):
            input = torch.tensor(input, dtype=torch.float32)
            input = input.squeeze(-1).unsqueeze(0)
        if self.training:
            return F.linear(input,
                            self.weight_mu + self.weight_sigma * self.weight_epsilon,
                            self.bias_mu + self.bias_sigma * self.bias_epsilon,)
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)
        

class TetrisActorCritic(nn.Module):
    '''
    Actor-Critic Network only designed for TetrisEnv. ~950K params.
    '''
    def __init__(self, input_size, output_size, use_smirl=False, noisy=False, epsilon=0.1):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.epsilon = epsilon

        linear = NoisyLinear if noisy else nn.Linear

        # state : image(1, 20, 20) / mino_pos(2,) / mino_rot(1,) / mino(one-hot) / hold(one-hot) / preview(one-hot * 5) / status(4,)
        self.img_feature = nn.Sequential(
            # input : (1, 20, 20)
            nn.Conv2d(1, 16, kernel_size=6, stride=2),  # (1, 8, 8, 16)
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=1), # (1, 5, 5, 32)
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1), # (1, 3, 3, 32)
            nn.LeakyReLU(),
            nn.Flatten(),                              # (1, 3 * 3 * 32)
            linear(3 * 3 * 32, 128),
            nn.ReLU(),
        )

        self.encode = linear(128 + 2, 32)
        self.mino_feature = linear(7 + 4, 32)
        self.hold_feature = linear(8, 32)
        self.preview_feature = linear(35, 32)
        self.mhp_feature = linear(32 + 32 + 32, 128)
        self.imhps_feature = linear(128 + 128 + 4, 256)
        #self.imhps_feature = linear(128 + 7 + 4 + 2 + 8 + 35 + 4, 256)

        self.actor = nn.Sequential(
            linear(256, 256),
            nn.ReLU(),
            linear(256, output_size)
        )

        self.extra_layer = nn.Sequential(
            linear(256, 256),
            nn.ReLU(),
        )

        self.critic_ext = linear(256, 1)
        self.critic_int = linear(256, 1)

        self.use_smirl = use_smirl
        if use_smirl:
            self.smirl_feature = nn.Sequential(
                linear(401, 256),
                nn.ReLU(),
                linear(256, 256),
            )
            for i in range(len(self.smirl_feature)):
                if type(self.smirl_feature[i]) == nn.Linear:
                    nn.init.orthogonal_(self.smirl_feature[i].weight, 0.1)
                    self.smirl_feature[i].bias.data.zero_()

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                m.bias.data.zero_()

        nn.init.orthogonal_(self.encode.weight, 0.1)
        self.encode.bias.data.zero_()
        nn.init.orthogonal_(self.mino_feature.weight, 0.1)
        self.mino_feature.bias.data.zero_()
        nn.init.orthogonal_(self.hold_feature.weight, 0.1)
        self.hold_feature.bias.data.zero_()
        nn.init.orthogonal_(self.preview_feature.weight, 0.1)
        self.preview_feature.bias.data.zero_()
        nn.init.orthogonal_(self.mhp_feature.weight, 0.01)
        self.mhp_feature.bias.data.zero_()
        nn.init.orthogonal_(self.imhps_feature.weight, 0.01)
        self.imhps_feature.bias.data.zero_()

        nn.init.orthogonal_(self.critic_ext.weight, 0.01)
        self.critic_ext.bias.data.zero_()
        nn.init.orthogonal_(self.critic_int.weight, 0.01)
        self.critic_int.bias.data.zero_()

        for i in range(len(self.actor)):
            if type(self.actor[i]) == nn.Linear:
                nn.init.orthogonal_(self.actor[i].weight, 0.01)
                self.actor[i].bias.data.zero_()

        for i in range(len(self.extra_layer)):
            if type(self.extra_layer[i]) == nn.Linear:
                nn.init.orthogonal_(self.extra_layer[i].weight, 0.1)
                self.extra_layer[i].bias.data.zero_()

    def best_state(self, states, grouped_actions):
        '''
        Use epsilon-greedy policy
        '''
        max_value = None
        best_state = None
        best_actions = None
        if random.random() <= self.epsilon:
            return random.choice(list(zip(states, grouped_actions)))
        else:
            for state, actions in zip(states, grouped_actions):
                _, ve, vi, _, _, _ = self.forward(np.array(state).reshape([1,self.input_size]))
                value = ve + vi
                if not max_value or value > max_value:
                    max_value = value
                    best_state = state
                    best_actions = actions

        return best_state, best_actions

    def forward(self, s):
        img, mino_pos, mino_rot, mino, hold, preview, status, smirl = parse_dict(s, self.use_smirl)
        #img, others, smirl = parse_dict_simple(s, self.use_smirl)
        
        img_feature = self.img_feature(img)
        mino_feature = F.relu(self.mino_feature(torch.cat((mino, mino_rot),dim=1)) + \
                              self.encode(torch.cat((img_feature, mino_pos),dim=1)))
        hold_feature = F.relu(self.hold_feature(hold))
        preview_feature = F.relu(self.preview_feature(preview))
        
        mhp_feature = F.relu(self.mhp_feature(
                        torch.cat((mino_feature, hold_feature, preview_feature),dim=1)
                        ))
        if self.use_smirl:
            imhps_feature = F.relu(self.imhps_feature(
                        torch.cat((img_feature, mhp_feature, status),dim=1)
                        ) + self.smirl_feature(smirl))
        else:
            imhps_feature = F.relu(self.imhps_feature(
                        torch.cat((img_feature, mhp_feature, status),dim=1)
                        ))
        #imhps_feature = F.relu(self.imhps_feature(
        #           torch.cat((img_feature, others),dim=1)
        #           ))

        result = F.softmax(self.actor(imhps_feature),dim=1)
        #if self.use_smirl:
        #    smirl_feature = self.smirl_feature(smirl)
        #    result = F.softmax(self.actor(imhps_feature + smirl_feature),dim=1)
        #else:
        #    result = F.softmax(self.actor(imhps_feature),dim=1)
        
        action_dist = Categorical(result)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action).unsqueeze(1)
        entropy = action_dist.entropy().unsqueeze(1)

        value_feed = self.extra_layer(imhps_feature) + imhps_feature
        value_ext = self.critic_ext(value_feed)
        value_int = self.critic_int(value_feed)

        return action.detach(), value_ext, value_int, \
               log_prob.detach(), entropy.detach(), result
    

class TetrisQ(nn.Module):
    '''
    Q Network only designed for TetrisEnv. ~950K params.
    '''
    def __init__(self, input_size, output_size, use_smirl=False, noisy=False, epsilon=0.1):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.epsilon = epsilon

        linear = NoisyLinear if noisy else nn.Linear

        # state : image(1, 20, 20) / mino_pos(2,) / mino_rot(1,) / mino(one-hot) / hold(one-hot) / preview(one-hot * 5) / status(4,)
        self.img_feature = nn.Sequential(
            # input : (1, 20, 20)
            nn.Conv2d(1, 16, kernel_size=6, stride=2),  # (1, 8, 8, 16)
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=1), # (1, 5, 5, 32)
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1), # (1, 3, 3, 32)
            nn.LeakyReLU(),
            nn.Flatten(),                              # (1, 3 * 3 * 32)
            linear(3 * 3 * 32, 128),
            nn.ReLU(),
        )

        self.encode = linear(128 + 2, 32)
        self.mino_feature = linear(7 + 4, 32)
        self.hold_feature = linear(8, 32)
        self.preview_feature = linear(35, 32)
        self.mhp_feature = linear(32 + 32 + 32, 128)
        self.imhps_feature = linear(128 + 128 + 4, 256)
        #self.imhps_feature = linear(128 + 7 + 4 + 2 + 8 + 35 + 4, 256)

        self.critic = nn.Sequential(
            linear(256, 256),
            nn.ReLU(),
            linear(256, 7)
        )

        self.use_smirl = use_smirl
        if use_smirl:
            self.smirl_feature = nn.Sequential(
                linear(401, 256),
                nn.ReLU(),
                linear(256, 256),
            )
            for i in range(len(self.smirl_feature)):
                if type(self.smirl_feature[i]) == nn.Linear:
                    nn.init.orthogonal_(self.smirl_feature[i].weight, 0.1)
                    self.smirl_feature[i].bias.data.zero_()

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                m.bias.data.zero_()

        nn.init.orthogonal_(self.encode.weight, 0.1)
        self.encode.bias.data.zero_()
        nn.init.orthogonal_(self.mino_feature.weight, 0.1)
        self.mino_feature.bias.data.zero_()
        nn.init.orthogonal_(self.hold_feature.weight, 0.1)
        self.hold_feature.bias.data.zero_()
        nn.init.orthogonal_(self.preview_feature.weight, 0.1)
        self.preview_feature.bias.data.zero_()
        nn.init.orthogonal_(self.mhp_feature.weight, 0.01)
        self.mhp_feature.bias.data.zero_()
        nn.init.orthogonal_(self.imhps_feature.weight, 0.01)
        self.imhps_feature.bias.data.zero_()

        for i in range(len(self.critic)):
            if type(self.critic[i]) == nn.Linear:
                nn.init.orthogonal_(self.critic[i].weight, 0.1)
                self.critic[i].bias.data.zero_()

    def best_state(self, states, grouped_actions):
        '''
        Use epsilon-greedy policy
        '''
        max_value = None
        best_state = None
        best_actions = None
        if random.random() <= self.epsilon:
            return random.choice(list(zip(states, grouped_actions)))
        else:
            for state, actions in zip(states, grouped_actions):
                value = self.forward(np.array(state).reshape([1,self.input_size])).sum()
                if not max_value or value > max_value:
                    max_value = value
                    best_state = state
                    best_actions = actions

        return best_state, best_actions

    def forward(self, s):
        img, mino_pos, mino_rot, mino, hold, preview, status, smirl = parse_dict(s, self.use_smirl)
        #img, others, smirl = parse_dict_simple(s, self.use_smirl)
        
        img_feature = self.img_feature(img.reshape([-1,1,20,20]))
        mino_feature = F.relu(self.mino_feature(torch.cat((mino.reshape([-1,7]), 
                                                           mino_rot.reshape([-1,4])),dim=1)) + \
                              self.encode(torch.cat((img_feature, mino_pos.reshape([-1,2])),dim=1)))
        hold_feature = F.relu(self.hold_feature(hold.reshape([-1,8])))
        preview_feature = F.relu(self.preview_feature(preview.reshape([-1,35])))
        
        mhp_feature = F.relu(self.mhp_feature(
                        torch.cat((mino_feature, hold_feature, preview_feature),dim=1)
                        ))
        if self.use_smirl:
            imhps_feature = F.relu(self.imhps_feature(
                        torch.cat((img_feature, mhp_feature, status.reshape([-1,4])),dim=1)
                        ) + self.smirl_feature(smirl.reshape([-1,401])))
        else:
            imhps_feature = F.relu(self.imhps_feature(
                        torch.cat((img_feature, mhp_feature, status.reshape([-1,4])),dim=1)
                        ))
        #imhps_feature = F.relu(self.imhps_feature(
        #            torch.cat((img_feature, others),dim=1)
        #            ))

        #result = F.softmax(self.actor(imhps_feature),dim=1)
        if self.use_smirl:
            smirl_feature = self.smirl_feature(smirl)
            result = self.critic(imhps_feature + smirl_feature)
        else:
            result = self.critic(imhps_feature)

        return result
    
