import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np
from utils import parse_dict

class RND(nn.Module):
    '''
    Linear Networks for RND
    '''
    def __init__(self, input_size, hidden_size, output_size, num_layers, activation):
        super().__init__()

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


class TetrisRandom(nn.Module):
    '''
    Random network for TetrisEnv
    '''
    def __init__(self, use_smirl=False):
        super().__init__()

        # state : image(1, 20, 20) / mino_pos(2,) / mino_rot(1,) / mino(one-hot) / hold(one-hot) / preview(one-hot * 5) / status(4,)
        self.img_feature = nn.Sequential(
            # input : (1, 20, 20)
            nn.Conv2d(1, 8, kernel_size=4, stride=1),  # (1, 17, 17, 8)
            nn.LeakyReLU(),
            nn.Conv2d(8, 32, kernel_size=2, stride=2), # (1, 8, 8, 32)
            nn.LeakyReLU(),
            nn.Flatten(),                              # (1, 8 * 8 * 32)
            nn.Linear(8 * 8 * 32, 128),
            nn.ReLU(),
        )

        self.encode = nn.Linear(128 + 2, 32)
        self.mino_feature = nn.Linear(7 + 4, 32)
        self.hold_feature = nn.Linear(8, 32)
        self.preview_feature = nn.Linear(35, 32)
        self.mhp_feature = nn.Linear(32 + 32 + 32, 128)
        self.imhps_feature = nn.Linear(128 + 128 + 4, 256)

        self.use_smirl = use_smirl
        if use_smirl:
            self.smirl_feature = nn.Sequential(
                nn.Linear(401, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
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

    def forward(self, s):
        img, mino_pos, mino_rot, mino, hold, preview, status, smirl = parse_dict(s, self.use_smirl)

        img_feature = self.img_feature(img)
        mino_feature = F.relu(self.mino_feature(torch.cat((mino, mino_rot),dim=1)) + \
                              self.encode(torch.cat((img_feature, mino_pos),dim=1)))
        hold_feature = F.relu(self.hold_feature(hold))
        preview_feature = F.relu(self.preview_feature(nn.Flatten()(preview)))

        mhp_feature = F.relu(self.mhp_feature(torch.cat((mino_feature, hold_feature, preview_feature),dim=1)))
        if self.use_smirl:
            imhps_feature = F.relu(self.imhps_feature(
                        torch.cat((img_feature, mhp_feature, status),dim=1)
                        ) + self.smirl_feature(smirl))
        else:
            imhps_feature = F.relu(self.imhps_feature(
                        torch.cat((img_feature, mhp_feature, status),dim=1)
                        ))

        return imhps_feature
    

class TetrisPredictor(TetrisRandom):
    '''
    Predictor network for TetrisEnv
    '''
    def __init__(self, use_smirl=False):
        super().__init__(use_smirl=use_smirl)

        self.critic = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )

        for i in range(len(self.critic)):
            if type(self.critic[i]) == nn.Linear:
                nn.init.orthogonal_(self.critic[i].weight, 0.01)
                self.critic[i].bias.data.zero_()

    def forward(self, s):
        value = self.critic(super().forward(s))

        return value


class TetrisRND(nn.Module):
    '''
    Random Network Distillation for TetrisEnv
    '''
    def __init__(self, use_smirl=False):
        super().__init__()

        self.predictor = TetrisPredictor(use_smirl)
        self.target = TetrisRandom(use_smirl)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.orthogonal_(m.weight, np.sqrt(2))
                m.bias.data.zero_()

        for param in self.target.parameters():
            param.requires_grad = True

    def forward(self, s):
        target_feature = self.target(s)
        predict_feature = self.predictor(s)

        return predict_feature, target_feature
    
    def eval_int(self, s):
        predict_next, target_next = self.forward(s)

        return (target_next - predict_next).pow(2).sum(1) / 2

    def save(self, save_dir, env_name):
        import os
        predictor_path = os.path.join(save_dir, env_name + '.pred')
        target_path = os.path.join(save_dir, env_name + '.target')
        torch.save(self.predictor.state_dict(), predictor_path)
        torch.save(self.target.state_dict(), target_path)