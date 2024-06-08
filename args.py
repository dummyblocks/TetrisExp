import argparse
import os
import torch

def get_args():
    parser = argparse.ArgumentParser(description='TetrisRL')
    parser.add_argument('--algo', default='p',
                        help='Algorithm to use: PPO(p) + (RND(r)) + (SMiRL(s))')
    parser.add_argument('--group-actions', action='store_true', default=False,
                        help='Use field-state-oriented training (default : False)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--num-worker', type=int, default=8,
                        help='Number of workers to use (default: 8)')
    parser.add_argument('--subproc', action='store_true', default=False,
                        help='Use multiprocessing for environments(default: False)')
    parser.add_argument('--num-step', type=int, default=4096,
                        help='Number of forward steps (default: 4096)')
    parser.add_argument('--eps', type=float, default=0.2,
                        help='Epsilon (default: 0.2)')
    parser.add_argument('--ext-gamma', type=float, default=0.999,
                        help='Extrinsic discount factor for rewards (default: 0.999)')
    parser.add_argument('--int-gamma', type=float, default=0.99,
                        help='Intrinsic discount factor for rewards (default: 0.99)')
    parser.add_argument('--use-gae', action='store_true', default=True,
                        help='use generalized advantage estimation (default: True)')
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="Lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='Use GPU training (default: False)')
    parser.add_argument('--noisy', action='store_true', default=False,
                        help='Use NoisyNet (default: False)')
    parser.add_argument('--epoch', type=int, default=4,
                        help='number of epochs (default: 4)')
    parser.add_argument('--num-episode', type=int, default=1250,
                        help='number of episodes (default: 1250)')
    parser.add_argument('--mini-batch', type=int, default=64,
                        help='Number of batches (default: 64)')
    parser.add_argument('--critic_coef', type=int, default=0.5,
                        help='critic term coefficient (default: 0.5)')
    parser.add_argument('--entropy-coef', type=float, default=0.001,
                        help='entropy term coefficient (default: 0.001)')
    parser.add_argument('--ext-coef', type=float, default=2.,
                        help='extrinsic reward coefficient (default: 2.)')    
    parser.add_argument('--int-coef', type=float, default=1.,
                        help='intrinsic reward coefficient (default: 1.)')
    parser.add_argument('--pre-obs-norm-steps', type=int, default=10,
                        help='Number of steps for pre-normalization (default: 10)')
    parser.add_argument('--save-interval', type=int, default=5,
                        help='Save interval, one save per n updates (default: 5)')
    parser.add_argument('--load-model', action='store_true', default=False,
                        help='Load pre-trained Model (default: False)')
    parser.add_argument('--log-dir', default=None,
                        help='Directory to save agent logs (default: logs/CURRENT_DATETIME_HOSTNAME)')
    parser.add_argument('--save-dir', default=None,
                        help='Directory to save agent logs (default: trained_models/ALGORITHM_CURRENT_DATETIME)')
    parser.add_argument('--render', action='store_true', default=False,
                        help='Render training (default : False)')
    parser.add_argument('--env-name', default='SinglePlayerTetris',
                        help='Environment to train on (default: SinglePlayerTetris)')

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.batch_size = int(args.num_step * args.num_worker / args.mini_batch)
    if args.log_dir is not None:
        args.log_dir = os.path.join("runs", args.log_dir)
    if args.save_dir is not None:
        args.save_dir = os.path.join("trained_models", args.save_dir)
    else:
        import datetime
        args.save_dir = os.path.join("trained_models", args.algo + "_{date:%Y-%m-%d_%H-%M}".format(date=datetime.datetime.now()))

    return args