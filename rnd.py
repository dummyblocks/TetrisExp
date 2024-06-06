import torch
import torch.nn as nn
from torch.nn import init
import numpy as np
from utils import parse_dict

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


class TetrisPredictor(nn.Module):
    '''
    Predict network for TetrisEnv
    '''
    def __init__(self):
        super().__init__()


class TetrisRandom(nn.Module):
    '''
    Random network for TetrisEnv
    '''
    def __init__(self, input_size, output_size, noisy=False):
        super().__init__()

        # state : image(1, 20, 20) / mino_pos(2,) / mino_rot(1,) / mino(one-hot) / hold(one-hot) / preview(one-hot * 5) / status(4,)
        self.img_feature = nn.Sequential(
            # input : (1, 20, 20)
            nn.Conv2d(1, 32, kernel_size=8),  # (1, 13, 13, 32)
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4), # (1, 10, 10, 64)
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3), # (1, 8, 8, 64)
            nn.LeakyReLU(),
            nn.Flatten(),                     # (1, 8 * 8 * 64)
            nn.Linear(8 * 8 * 64, 256),
            nn.ReLU(),
        )

        self.mino_feature = nn.Linear(8 + 4, 32)
        self.hold_feature = nn.Linear(8, 32)
        self.preview_feature = nn.Linear(8, 32)
        self.mhp_feature = nn.Linear(32 + 32 + 32, 256)
        self.imhp_feature = nn.Linear(256 + 256, 512 + 4)

        self.actor = nn.Sequential(
            nn.Linear(512 + 4, 512),
            nn.ReLU(),
            nn.Linear(512, 7)
        )

        self.extra_layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
        )

        self.critic_ext = nn.Linear(512, 1)
        self.critic_int = nn.Linear(512, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                m.bias.data.zero_()

        nn.init.orthogonal_(self.mino_feature.weight, 0.1)
        self.critic.bias.data.zero_()
        nn.init.orthogonal_(self.hold_feature.weight, 0.1)
        self.critic.bias.data.zero_()
        nn.init.orthogonal_(self.preview_feature.weight, 0.1)
        self.critic.bias.data.zero_()
        nn.init.orthogonal_(self.imhp_feature.weight, 0.01)
        self.critic.bias.data.zero_()

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
        
        for i in range(len(self.mhp_feature)):
            if type(self.mhp_feature[i]) == nn.Linear:
                nn.init.orthogonal_(self.mhp_feature[i].weight, 0.01)
                self.mhp_feature[i].bias.data.zero_()

    def encode(self, coord, dim):

        return 

    def forward(self, s):
        img, mino_pos, mino_rot, mino, hold, preview, status = parse_dict(s)

        img_feature = self.img_feature(img)
        mino_feature = F.relu(self.mino_feature(torch.cat((mino, mino_rot)))) + self.encode(mino_pos, 32)
        hold_feature = F.relu(self.hold_feature(hold))
        preview_feature = F.relu(self.preview_feature(nn.Flatten()(preview)))
        mhp_feature = F.relu(self.mhp_feature(torch.cat((mino_feature, hold_feature, preview_feature))))
        imhp_feature = F.relu(self.imhp_feature(torch.cat((img_feature, mhp_feature))))

        result = F.softmax(self.actor(torch.cat(imhp_feature, status)))
        action_dist = Categorical(result)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action).unsqueeze(1)
        entropy = action_dist.entropy().unsqueeze(1)

        value_feed = self.extra_layer(imhp_feature)+imhp_feature
        value_ext = self.critic_ext(value_feed)
        value_int = self.critic_int(value_feed)

        return value


class TetrisRND(nn.Module):
    '''
    Random Network for TetrisEnv
    '''
    def __init__(self):
        super(RND, self).__init__()

        self.predictor = TetrisPredictor()
        self.target = TetrisRandom()

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