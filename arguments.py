import argparse
import torch


def argparser():
    parser = argparse.ArgumentParser(description='SAC-Batch-Env')
    # argument for random obstacles
    parser.add_argument('--rand_obs', type=bool, default=True,
                        help='randomize obstacles position (default: True)')
    # argument for numerous actor
    parser.add_argument('--n_envs', type=int, default=32,
                        help='Number of environments (default: 16)')
    # arguments for sac
    parser.add_argument('--eval', type=bool, default=True,
                        help='Evaluates a policy a policy every 10 episode (default: True)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                                term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                        help='Automaically adjust α (default: True)')
    parser.add_argument('--updates_per_collection', type=int, default=16, metavar='N',
                        help='model updates per simulator step (default: 16)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                        help='Steps sampling random actions (default: 10000)')
    parser.add_argument('--target_update_interval', type=int, default=5, metavar='N',
                        help='Value target update per no. of updates per step (default: 5)')
    # argument for replay memory
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                        help='random seed (default: 123456)')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='batch size (default: 128)')
    parser.add_argument('--num_steps', type=int, default=3000001, metavar='N',
                        help='maximum number of steps (default: 3000000)')
    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')
    # argument for neural network
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--cuda', action="store_true",
                        help='run on CUDA (default: False)')

    args = parser.parse_args()

    return args
