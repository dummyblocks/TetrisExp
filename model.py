import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

class ActorNN(nn.Module):
    '''
    A linear discrete policy network
    '''
    def __init__(self, input_size, hidden_size, output_size, num_layers, activation, noisy):
        super(ActorNN, self).__init__()
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
        super(CriticNN, self).__init__()

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