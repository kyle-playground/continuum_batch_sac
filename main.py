from continuum_robot_env import ContinuumRobotEnv
import numpy as np
import os
import torch
from sac import SAC
from torch.utils.tensorboard import SummaryWriter
import datetime
from arguments import argparser
from memory import ReplayMemory
from multiprocessing import Value, Queue, Process


args = argparser()


def bring_episode_exp(agent, rand_exploring, process_seed, total_numsteps, total_episode, trans_queue):

    # create a agent for each process
    env = ContinuumRobotEnv(seed=process_seed, random_obstacle=args.rand_obs)
    episode_steps = 0
    done = False
    state = env.reset()
    transitions_batch = []

    while not done:

        if rand_exploring:
            action = env.random_move()  # Sample random action
        else:
            action = agent.select_action(state)   # Sample action from policy

        next_state, reward, done, info = env.step(action)  # Step

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == env.MAX_STEP else float(not done)
        transitions = [state, action, reward, next_state, mask]

        transitions_batch.append(transitions)
        episode_steps += 1
        state = next_state

        if done:
            trans_queue.put(transitions_batch)
            with total_episode.get_lock():
                total_episode.value += 1

            with total_numsteps.get_lock():
                total_numsteps.value += episode_steps



def run():
    # writer
    writer = SummaryWriter('runs/{}_SAC_{}envs'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                                                       "1" if not args.rand_obs else args.n_envs))
    # set seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    process_seeds = [(i+1)*16 for i in range(args.n_envs)]

    # Agent
    env = ContinuumRobotEnv()
    state_dim = env.observation_space()
    if torch.cuda.is_available():
        agent = SAC(state_dim, env, args, gpu=True)
    else:
        agent = SAC(state_dim, env, args)
    env.close()

    # Memory
    memory = ReplayMemory(args.replay_size, args.seed)

    # Other variables for training process
    total_numsteps = Value('i', 0)
    total_episode = Value('i', 0)
    test_num = 0
    updates = 0
    rand_exploring = True

    while True:
        # get memory from multi env
        if total_numsteps.value > args.start_steps:
            rand_exploring = True

        # Arguments for child processes
        processes = []
        trans_queue = Queue()
        agent_cpu = agent
        agent_cpu.mv2cpu()
        # Send agent to Batch Environments and get batch exps
        for n in range(args.n_envs):
            p = Process(target=bring_episode_exp, args=(agent_cpu, rand_exploring, process_seeds[n],
                                                            total_numsteps, total_episode, trans_queue))
            p.start()
            processes.append(p)
        for process in processes:
            process.join(timeout=0.01)  # Not sure why processes fail to join without timeout use timeout to skip
            transitions_batch = trans_queue.get()
            memory.push_batch(transitions_batch)
        del processes
        del agent_cpu

        if len(memory) > args.batch_size:
            breakpoint()
            for _ in range(args.updates_per_collection):

                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory,
                                                                                                     args.batch_size,
                                                                                                     updates)
                updates += 1

        if total_numsteps.value > args.num_steps:
            break

        if total_episode.value - 100*(test_num+1) > 0 and args.eval is True:
            test_num += 1
            avg_reward = 0.
            episodes = 10
            num_success = 0
            env = ContinuumRobotEnv(seed=total_episode.value, random_obstacle=True)

            for _ in range(episodes):
                episode_reward = 0.
                done = False
                state = env.reset()
                info = "Go~~~"
                while not done:
                    action = agent.select_action(state, evaluate=True)
                    next_state, reward, done, info = env.step(action)
                    episode_reward += reward
                    state = next_state
                avg_reward += episode_reward

                if info == "is success":
                    num_success += 1

            env.close()
            avg_reward /= episodes

            print("----------------------------------------")
            print("num update: {}".format(updates))
            print("Total Episodes: {}, Total num_steps: {}".format(total_episode.value, total_numsteps.value))
            print("Test Episodes: {}, Avg. Reward: {}, Success rate: {}".format(episodes, round(avg_reward, 2), num_success/10))
            print("----------------------------------------")
            writer.add_scalar('Testing Avg reward', avg_reward, total_episode.value)
            writer.add_scalar('Success rate', num_success/10, total_episode.value)

    # agent.save_model("SAC_multiEnv")


if __name__ == '__main__':
    run()
