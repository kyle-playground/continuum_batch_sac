from continuum_robot_env import ContinuumRobotEnv
import time
env = ContinuumRobotEnv(use_GUI=True, random_obstacle=True, seed=15)
env.reset()
for _ in range(10000):
    env.step(env.random_move())
    env.visualize_goal(env.goal)
    time.sleep(1/200.)


