# SAC adapted from https://github.com/pranz24/pytorch-soft-actor-critic

from continuum_robot_env import ContinuumRobotEnv
import numpy as np
import itertools
import torch
from sac import SAC
from memory import ReplayMemory
from arguments import argparser
from multiprocessing import Process, Queue
from test import test_env
from utils import hard_update
from torch.utils.tensorboard import SummaryWriter
import datetime

args = argparser()

# Environment
env = ContinuumRobotEnv()
env.reset()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Agent
state_dim = env.observation_space()
agent = SAC(state_dim, env, args, gpu=True)
agent_cpu = SAC(state_dim, env, args)


# Tesnorboard

writer = SummaryWriter('runs/{}_SAC_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), "1_env",
                                   "autotune" if args.automatic_entropy_tuning else ""))

# Memory
memory = ReplayMemory(args.replay_size, args.seed)

# Training Loop
total_numsteps = 0
updates = 0


for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()

    while not done:
        if args.start_steps > total_numsteps:
            action = env.random_move()  # Sample random action
        else:
            action = agent.select_action(state)  # Sample action from policy

        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)

                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                updates += 1

        next_state, reward, done, info = env.step(action)  # Step

        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == env.MAX_STEP else float(not done)

        memory.push(state, action, reward, next_state, mask) # Append transition to memory

        state = next_state

    if total_numsteps > args.num_steps:
        break

    if i_episode % 100 == 0 and args.eval is True:
        hard_update(agent_cpu.policy, agent.policy)
        reward_queue = Queue(maxsize=1)
        suc_queue = Queue(maxsize=1)
        p = Process(target=test_env, args=(i_episode, total_numsteps, updates, agent_cpu, reward_queue, suc_queue))
        p.start()
        p.join()
        avg_reward = reward_queue.get()
        num_success = suc_queue.get()

        writer.add_scalar('Testing Avg reward', avg_reward, i_episode)
        writer.add_scalar('Success rate', num_success / 10, i_episode)


agent.save_model("SAC_trained_with_1_env")
env.close()

