import pybullet as p
import pybullet_data
import numpy as np

START_POS = [0, 0, 0]
START_ORIENTATION = p.getQuaternionFromEuler([0, 0, 0])
CYLINDER_R_H = [0.25, 10.]   # TODO: real scale is required to modify
SPHERE_R = 0.1
BOX_LWH = [3., 3., 1.5]       # TODO: real scale is required to modify
MASS = 0
visualShapeId = -1
# TODO: real scale is required to modify
# OBSTACLE_POS_8 = [[8., -3., 0.], [8., -1., 0.], [8., 1., 0.], [8., 3., 0.],
#                 [9.5, -4., 0.], [9.5, -2., 0.], [9.5, 0., 0.], [9.5, 2., 0.], [9.5, 4., 0.]]
OBSTACLE_POS = [[8., -2.5, 0.], [8., -1., 0.], [8., 1., 0.], [8., 2.5, 0.]]

OBSTACLE_ORIENTATION = p.getQuaternionFromEuler([0, 0, 0])
TRACK_POS = [3.5, 0., 0.]     # TODO: real scale is required to modify
TRACK_ORIENTATION = p.getQuaternionFromEuler([0, 0, 0])
# DISTANT_THRESHOLD = 0.1       # TODO: real scale is required to modify


class ContinuumRobotEnv:
    def __init__(self, seed=0, use_GUI=False, random_obstacle=False):
        self.MAX_STEP = 30
        self.step_count = 0
        self.reward_scalar = 3
        self.goal = None
        self.pos = None
        self.contact = False
        self.contact_info = ""

        np.random.seed(seed)

        if use_GUI:
            self.physicsClient = p.connect(p.GUI)
            print("physics server with GUI connected")
        else:
            self.physicsClient = p.connect(p.DIRECT)
            print("physics server connected")

        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        # load plane and robot, and build obstacles
        self.planeId = p.loadURDF("plane.urdf")
        self.robotId = p.loadURDF("manipulator/robot.xml", START_POS, START_ORIENTATION, useFixedBase=True)
        col_track_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=BOX_LWH)
        self.track_Id = p.createMultiBody(MASS, col_track_id, visualShapeId, TRACK_POS, TRACK_ORIENTATION)

        if not random_obstacle:
            self.obstacle_pos = np.array(OBSTACLE_POS)
        else:
            self.obstacle_pos = self.random_obstacle_pos()

        col_cylinder_id = p.createCollisionShape(p.GEOM_CYLINDER, radius=CYLINDER_R_H[0], height=CYLINDER_R_H[1])
        for i in range(0, len(self.obstacle_pos)):
            cylinder_Id = p.createMultiBody(MASS, col_cylinder_id, visualShapeId, self.obstacle_pos[i],
                                            OBSTACLE_ORIENTATION)

        self.num_joints = p.getNumJoints(self.robotId)
        self.num_real_links = int(self.num_joints/2)
        self.num_bodies = p.getNumBodies(self.physicsClient)
        joint_info_pri = p.getJointInfo(self.robotId, 1)
        joint_info_rev = p.getJointInfo(self.robotId, 2)
        self.constraint_pri = np.array([joint_info_pri[8], joint_info_pri[9]])
        self.constraint_rev = np.array([joint_info_rev[8], joint_info_rev[9]])

    def step(self, delta_pos):

        info = " Not Yet "
        done = False
        self.step_count += 1
        self.pos += delta_pos

        self.pos[0] = self.pos[0].clip(self.constraint_pri[0], self.constraint_pri[1])
        self.pos[1:] = self.pos[1:].clip(self.constraint_rev[0], self.constraint_rev[1])

        motor_pos = np.insert(self.pos, 0, 0)
        p.setJointMotorControlArray(self.robotId, range(self.num_joints), controlMode=p.POSITION_CONTROL,
                                    targetPositions=motor_pos)

        for i in range(25):
            p.stepSimulation()

        observation, final_pos, min_dis_from_obs = self.get_observation()
        observation = np.concatenate((observation, self.goal))
        distance = np.linalg.norm(final_pos - self.goal)
        reward = -distance * self.reward_scalar

        contact_list = []
        for i in range(0, self.num_real_links):
            index = 2 * i + 1
            contact_list.append(p.getContactPoints(self.robotId, index))
        contact = True in [True if x else False for x in contact_list]

        if contact:
            reward += -2 * self.reward_scalar
            self.contact = True
            self.contact_info = "   Warning!!!"
            # done = True   # Experiment (if terminated when collide)
            # reward += -300

        if distance < 0.3 or self.step_count == self.MAX_STEP:
            done = True
            if distance < 0.3:
                if not self.contact:
                    reward += 5 * self.reward_scalar
                reward += 2 * self.reward_scalar
                info = "is success"
        info = info + self.contact_info

        return observation, reward, done, info

    def reset(self):
        self.step_count = 0
        self.contact = False
        self.contact_info = ""
        self.goal = self.pick_goal()
        self.pos = np.zeros(self.num_joints-1)

        for i in range(0, self.num_joints):
            p.resetJointState(self.robotId, i, 0)

        reset_state, _, _ = self.get_observation()
        reset_state = np.concatenate((reset_state, self.goal))

        return reset_state

    def get_observation(self):
        links_global_pos = []
        links_obs_dis = []
        links_rel_obs = []
        for i in range(0, self.num_real_links):
            index = 2*i + 1
            links_states = p.getLinkState(self.robotId, index)

            # add global position into observation
            links_global_pos.append(links_states[0])
            for j in range(4):
                # observation for the relationship between obstacles and links
                # distance
                links_obs_dis.append(np.linalg.norm(links_states[0][:2] - self.obstacle_pos[j][:2]))

                # relative position
                # links_observation.append(links_states[0][:2] - self.obstacle_pos[j][:2])
                links_rel_obs.append(links_states[0] - self.obstacle_pos[j])

        end2goal_pos = links_states[0] - self.goal

        # calculate the closest distance between robot and the nearest obstacle
        links_obs_dis = np.array(links_obs_dis)
        min_dis_from_obs = min(links_obs_dis)

        links_rel_obs = np.ravel(links_rel_obs)
        links_global_pos = np.ravel(links_global_pos)

        observation = np.concatenate((links_global_pos, links_rel_obs, end2goal_pos), axis=0)

        return observation, links_states[0], min_dis_from_obs

    def close(self):
        p.disconnect(self.physicsClient)
        print('Environment close')

    def random_move(self):
        random_pri = np.random.uniform(-0.5, 0.5, 1)
        random_rev = np.random.uniform(-0.05, 0.05, self.num_joints-2)
        random_pos = np.concatenate((random_pri, random_rev))
        return random_pos

    def observation_space(self):
        observation = self.reset()
        shape_obsv = len(observation)

        return shape_obsv

    def action_space(self):
        # action space = num_joint -1 (the fixed joint is excluded)
        shape_u = self.num_joints - 1

        low = np.insert(np.ones(shape_u - 1) * -0.05, 0, -0.5)
        high = np.insert(np.ones(shape_u - 1) * 0.05, 0, 0.5)

        return shape_u, high, low

    def visualize_goal(self, goal_pos):
        vis_goal_id = p.createVisualShape(p.GEOM_SPHERE, radius=SPHERE_R, rgbaColor=[1,0,0,1]) # red
        goalId = p.createMultiBody(MASS, baseVisualShapeIndex=vis_goal_id, basePosition=goal_pos)
        return goalId

    def pick_goal(self):
        # pick a goal from defined area
        random_pos_x = np.random.uniform(8.8, 9.3)
        random_pos_y = np.random.uniform(-3.5, 3.5)
        random_pos_z = np.random.uniform(2, 4)
        goal_pos = np.array([random_pos_x, random_pos_y, random_pos_z])
        return goal_pos

    def random_obstacle_pos(self):
        # randomly pick four positions and the distance between either two of them should larger than the radius

        # while True:
        #     random_pos_x = np.random.uniform(7.5, 8.5, 4).reshape(-1, 1)
        #     random_pos_y = np.random.uniform(-3, 3, 4).reshape(-1, 1)
        #     random_pos_z = np.zeros((4, 1))
        #     obstacles_pos = np.concatenate((random_pos_x, random_pos_y, random_pos_z), axis=1)
        #     A = np.linalg.norm(obstacles_pos[0]-obstacles_pos[1]) > 2*CYLINDER_R_H[0]
        #     B = np.linalg.norm(obstacles_pos[0]-obstacles_pos[2]) > 2*CYLINDER_R_H[0]
        #     C = np.linalg.norm(obstacles_pos[0]-obstacles_pos[3]) > 2*CYLINDER_R_H[0]
        #     D = np.linalg.norm(obstacles_pos[1]-obstacles_pos[2]) > 2*CYLINDER_R_H[0]
        #     E = np.linalg.norm(obstacles_pos[1]-obstacles_pos[3]) > 2*CYLINDER_R_H[0]
        #     F = np.linalg.norm(obstacles_pos[2]-obstacles_pos[3]) > 2*CYLINDER_R_H[0]
        #     if A and B and C and D and E and F:
        #         break
        obstacles_pos = np.array(OBSTACLE_POS)
        for i in range(len(obstacles_pos)):
            obstacles_pos[i][0] = obstacles_pos[i][0] + np.random.uniform(-0.5, 0.5)
            obstacles_pos[i][1] = obstacles_pos[i][1] + np.random.uniform(-0.25, 0.25)

        return obstacles_pos

