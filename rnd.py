import torch
import torch.nn as nn
from torch.nn import init
import numpy as np

class RND(nn.Module):
    '''
    Linear Networks for RND
    '''
    def __init__(self, input_size, hidden_size, output_size, num_layers, activation):
        super(RND, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.predictor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            activation,
            *((nn.Linear(hidden_size, hidden_size),
            activation) * (num_layers-2)),
            nn.Linear(hidden_size,output_size),
            activation,
            nn.Linear(output_size,output_size),
            activation,
            nn.Linear(output_size,output_size)
        )
        self.target = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            activation,
            *((nn.Linear(hidden_size, hidden_size),
            activation) * (num_layers-2)),
            nn.Linear(hidden_size,output_size)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.orthogonal_(m.weight, np.sqrt(2))
                m.bias.data.zero_()

        for param in self.target.parameters():
            param.requires_grad = True

    def forward(self, s):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, dtype=torch.float32)

        target_feature = self.target(s)
        predict_feature = self.predictor(s)

        return predict_feature, target_feature
    
    def eval_int(self, s):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, dtype=torch.float32)

        predict_next, target_next = self.forward(s)

        return (target_next - predict_next).pow(2).sum(1) / 2

    def save(self, save_dir, env_name):
        import os
        predictor_path = os.path.join(save_dir, env_name + '.pred')
        target_path = os.path.join(save_dir, env_name + '.target')
        torch.save(self.predictor.state_dict(), predictor_path)
        torch.save(self.target.state_dict(), target_path)