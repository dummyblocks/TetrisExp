import os
from args import get_args
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

from ppo import PPO
from async_env import Envs
from model import ActorNN, CriticNN
from rnd import RND
from smirl import SMiRL
import numpy as np

from env import SinglePlayerTetris

from tensorboardX import SummaryWriter

if __name__ == "__main__":
    args = get_args()
    device = torch.device('cuda' if args.cuda else 'cpu')

    print('Load environment..')

    if args.env_name == 'SinglePlayerTetris':
        env = SinglePlayerTetris()
        env = gym.wrappers.FlattenObservation(env)
    else:
        env = gym.make(args.env_name)
    input_size = env.observation_space.shape[0]
    hidden_size = 512
    num_layers = 5
    output_size = env.action_space.n

    smirl_size = env.get_wrapper_attr('w') * env.get_wrapper_attr('h')
    
    if 's' in args.algo:
        # add smirl state in input
        input_size += smirl_size + 1

    env.close()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    writer = SummaryWriter(log_dir=args.log_dir)

    envs = Envs(env, args.env_name, args.num_worker)

    print('Environment loading done..')

    actor = ActorNN(
        input_size, hidden_size, output_size, num_layers, nn.Tanh()
    )
    critic = CriticNN(
        input_size, hidden_size, 1, num_layers, nn.Tanh(),
        use_noisy=args.use_noisy_net, use_int='r' in args.algo
    )
    agent = PPO(
        actor=actor,
        critic=critic,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        epsilon=args.eps,
        episode_num=args.num_episode,
        episode_len=args.num_step,
        norm_len=args.pre_obs_norm_steps,
        batch_size=args.mini_batch,
        #memory_size=args.memory_size,
        lamda=args.gae_lambda,
        gamma_ext=args.ext_gamma,
        gamma_int=args.int_gamma,
        v_coef=args.critic_coef,
        ent_coef=args.entropy_coef,
        ext_coef=args.ext_coef,
        int_coef=args.int_coef,
        epochs=args.epoch,
        use_rnd='r' in args.algo,
        rnd_arg=(input_size, hidden_size, output_size, num_layers, nn.ReLU()),
        use_smirl='s' in args.algo,
        smirl_arg=smirl_size
    )

    if args.load_model:
        agent.load(args.save_dir, args.env_name, args.cuda)

    print('Model created..')
    print('Start training..')
    
    agent.train(envs, args.save_dir, args.save_interval, args.render, writer)