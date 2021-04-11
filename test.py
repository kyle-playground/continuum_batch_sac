from continuum_robot_env import ContinuumRobotEnv
from arguments import argparser
args = argparser()


def test_env(i_episode, total_numsteps, updates, agent, r_queue, s_queue):
    avg_reward = 0.
    episodes = 10
    num_success = 0

    env_test = ContinuumRobotEnv(seed=i_episode, random_obstacle=True)

    for _ in range(episodes):
        episode_reward = 0.
        done = False
        state = env_test.reset()
        info = "GO~~"
        while not done:

            action = agent.select_action(state, evaluate=True)
            next_state, reward, done, info = env_test.step(action)

            episode_reward += reward
            state = next_state

        avg_reward += episode_reward
        if info == "is success":
            num_success += 1

    env_test.close()
    avg_reward /= episodes

    r_queue.put(avg_reward)
    s_queue.put(num_success)

    print("----------------------------------------")
    print("num update: {}".format(updates))
    print("Total Episodes: {}, Total num_steps: {}".format(i_episode, total_numsteps))
    print("Test Episodes: {}, Avg. Reward: {}, Success rate: {}".format(episodes, round(avg_reward, 2),
                                                                        num_success / 10))
    print("----------------------------------------")
