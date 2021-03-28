from continuum_robot_env import ContinuumRobotEnv
import numpy as np
import os
import torch
from sac import SAC
from torch.utils.tensorboard import SummaryWriter
import datetime
from arguments import argparser
from memory import ReplayMemory
from multiprocessing import Value, Pipe, Queue, Process
import queue

args = argparser()


def bring_episode_exp(param_queue, conn, total_numsteps, total_episode, num):
    # set seeds for children processes
    process_seed = args.seed + num*args.n_actors
    torch.manual_seed(process_seed)
    np.random.seed(process_seed)

    # create a agent for each process
    env = ContinuumRobotEnv(seed=process_seed, random_obstacle=args.rand_obs)
    state_dim = env.observation_space()

    agent = SAC(state_dim, env, args)
    exploring_step = 0

    while True:
        episode_steps = 0
        done = False
        state = env.reset()

        while not done:

            if args.start_steps > exploring_step:
                action = env.random_move()  # Sample random action
                exploring_step += 1
            else:
                action = agent.select_action(state)   # Sample action from policy

            next_state, reward, done, info = env.step(action)  # Step

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            mask = 1 if episode_steps == env.MAX_STEP else float(not done)
            transitions = [state, action, reward, next_state, mask]

            conn.send(transitions)  # Append transition to memory
            episode_steps += 1
            state = next_state

            if done:

                with total_episode.get_lock():
                    total_episode.value += 1
                with total_numsteps.get_lock():
                    total_numsteps.value += episode_steps

                try:
                    param = param_queue.get(block=False)
                    agent.policy.load_state_dict(param)
                except queue.Empty:
                    pass

        if total_numsteps.value > args.num_steps:
            break


def run():
    # writer
    writer = SummaryWriter('runs/{}_SAC_{}workers_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                                                             args.n_actors, "rand" if args.rand_obs else "uniform"))
    # set seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Environment
    env = ContinuumRobotEnv()

    # Agent
    state_dim = env.observation_space()
    if torch.cuda.is_available():
        agent = SAC(state_dim, env, args, gpu=True)
    else:
        agent = SAC(state_dim, env, args)
    env.close()

    # Memory
    memory = ReplayMemory(args.replay_size, args.seed)
    receiver, sender = Pipe()
    param_queue = Queue(maxsize=args.n_actors+1)

    # Other variables for training process
    total_numsteps = Value('i', 0)
    total_episode = Value('i', 0)
    process = []
    test_num = 0
    updates = 0

    # get memory from multi env
    for num in range(args.n_actors):
        procs = Process(target=bring_episode_exp, args=(param_queue, sender,
                                                        total_numsteps, total_episode, num))
        procs.start()
        process.append(procs)

    while True:

        for _ in range(args.n_actors):
            transitions = receiver.recv()
            memory.push(transitions)

        if len(memory) > args.batch_size:
            for _ in range(args.updates_per_step):

                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory,
                                                                                                     args.batch_size,
                                                                                                     updates)
                updates += 1
            if param_queue.full():
                drop = param_queue.get()
            state_dict = agent.policy.state_dict()
            for k, v in state_dict.items():
                state_dict[k] = v.cpu()
            param_queue.put(state_dict)
            del state_dict

        if total_numsteps.value > args.num_steps:
            for p in process:
                p.terminate()
            break

        if total_episode.value - 10000*(test_num+1) > 0 and args.eval is True:
            test_num += 1
            avg_reward = 0.
            episodes = 10
            num_success = 0
            env = ContinuumRobotEnv(seed=total_episode.value, random_obstacle=True)

            for _ in range(episodes):
                episode_reward = 0.
                done = False
                state = env.reset()
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
