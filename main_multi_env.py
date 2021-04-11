from continuum_robot_env import ContinuumRobotEnv
import numpy as np
import torch
from sac import SAC
from memory import ReplayMemory
from arguments import argparser

from torch.utils.tensorboard import SummaryWriter
import datetime

args = argparser()

# Environments
Envs = []
for n in range(args.n_envs):
    if n == 0:
        CB_env = ContinuumRobotEnv()
    else:
        CB_env = ContinuumRobotEnv(seed=16*n, random_obstacle=args.rand_obs)
    CB_env.reset()
    Envs.append(CB_env)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Agent
env = Envs[0]
state_dim = env.observation_space()
agent = SAC(state_dim, env, args, gpu=True)

# Tesnorboard
writer = SummaryWriter('runs/{}_SAC_{}envs_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                                                      args.n_envs,
                                                      "autotune" if args.automatic_entropy_tuning else ""))

# Memory
memory = ReplayMemory(args.replay_size, args.seed)

# Training Loop
total_numsteps = 0
updates = 0
total_episode = 0
test_num = 0

# Initialization
episode_reward = [0 for i in range(args.n_envs)]
episode_steps = [0 for i in range(args.n_envs)]
state = [0 for i in range(args.n_envs)]
action = [0 for i in range(args.n_envs)]
for n in range(args.n_envs):
    state[n] = Envs[n].reset()

while True:

    if args.start_steps > total_numsteps:
        action = [env.random_move() for i in range(args.n_envs)]  # Sample random action
    else:
        action = [agent.select_action(state[i]) for i in range(args.n_envs)]  # Sample action from policy

    if len(memory) > args.batch_size:
        # Number of updates per step in environment
        for i in range(args.updates_per_collection):
            # Update parameters of all the networks
            critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)

            # writer.add_scalar('loss/critic_1', critic_1_loss, updates)
            # writer.add_scalar('loss/critic_2', critic_2_loss, updates)
            # writer.add_scalar('loss/policy', policy_loss, updates)
            # writer.add_scalar('loss/entropy_loss', ent_loss, updates)
            # writer.add_scalar('entropy_temprature/alpha', alpha, updates)
            updates += 1

    for n in range(args.n_envs):
        next_state, reward, done, info = Envs[n].step(action[n])  # Step
        episode_steps[n] += 1
        episode_reward[n] += reward
        total_numsteps += 1
        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps[n] == Envs[n].MAX_STEP else float(not done)

        transitions = [state[n], action[n], reward, next_state, mask]
        memory.push(transitions)  # Append transition to memory
        state[n] = next_state
        if done:
            total_episode += 1
            state[n] = Envs[n].reset()
            episode_steps[n] = 0
            episode_reward[n] = 0

    if total_numsteps > args.num_steps:
        break

    if total_episode - 100*(test_num+1) > 0 and args.eval is True:
        test_num += 1
        avg_reward = 0.
        episodes = 10
        num_success = 0
        Test_env = ContinuumRobotEnv(seed=total_episode, random_obstacle=True)

        for _ in range(episodes):

            test_episode_reward = 0.
            test_done = False
            test_state = Test_env.reset()
            info = "Go~~~"
            while not test_done:
                test_action = agent.select_action(test_state, evaluate=True)
                next_state, reward, test_done, info = Test_env.step(test_action)
                test_episode_reward += reward
                test_state = next_state
            avg_reward += test_episode_reward

            if info == "is success":
                num_success += 1

        Test_env.close()
        avg_reward /= episodes

        print("----------------------------------------")
        print("num update: {}".format(updates))
        print("Total Episodes: {}, Total num_steps: {}".format(total_episode, total_numsteps))
        print("Test Episodes: {}, Avg. Reward: {}, Success rate: {}".format(episodes, round(avg_reward, 2),
                                                                            num_success / 10))
        print("----------------------------------------")
        writer.add_scalar('Testing Avg reward', avg_reward, total_episode)
        writer.add_scalar('Success rate', num_success / 10, total_episode)


agent.save_model("SAC_trained_in_main_16_env")
for n in range(args.n_envs):
    Envs[n].close()

