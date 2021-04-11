import pybullet
import pybullet_data
import numpy as np
import pybullet_utils.bullet_client as bc

START_POS = [0, 0, 0]
START_ORIENTATION = pybullet.getQuaternionFromEuler([0, 0, 0])
CYLINDER_R_H = [0.25, 10.]   # TODO: real scale is required to modify
SPHERE_R = 0.1
BOX_LWH = [3., 3., 1.5]       # TODO: real scale is required to modify
MASS = 0
visualShapeId = -1

OBSTACLE_POS = [[8., -2.5, 0.], [8., -1., 0.], [8., 1., 0.], [8., 2.5, 0.]]  # TODO: real scale is required to modify
OBSTACLE_ORIENTATION = pybullet.getQuaternionFromEuler([0, 0, 0])
TRACK_POS = [3.5, 0., 0.]     # TODO: real scale is required to modify
TRACK_ORIENTATION = pybullet.getQuaternionFromEuler([0, 0, 0])


class ContinuumRobotEnv:
    def __init__(self, seed=0, activate_gui=False, random_obstacle=False):
        self.MAX_STEP = 30
        self.DISTANT_THRESHOLD = 0.3
        self.REWARD_SCALAR = 3

        self.step_count = 0
        self.goal = None
        self.pos = None
        self.contact = False
        self.contact_info = ""
        self.contact_num = 0

        np.random.seed(seed)

        if activate_gui:
            # self.physicsClient = p.connect(p.GUI)
            print("physics server with GUI connected")
        else:
            self.p_env = bc.BulletClient(connection_mode=pybullet.DIRECT)
            # self.physicsClient = p.connect(p.DIRECT)
            # print("physics server connected")

        self.p_env.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        # load plane and robot, and build obstacles
        self.planeId = self.p_env.loadURDF("plane.urdf")
        self.robotId = self.p_env.loadURDF("manipulator/robot.xml", START_POS, START_ORIENTATION, useFixedBase=True)
        col_track_id = self.p_env.createCollisionShape(self.p_env.GEOM_BOX, halfExtents=BOX_LWH)
        self.track_Id = self.p_env.createMultiBody(MASS, col_track_id, visualShapeId, TRACK_POS, TRACK_ORIENTATION)

        # Set obstacle position
        if not random_obstacle:
            self.obstacle_pos = np.array(OBSTACLE_POS)
        else:
            self.obstacle_pos = self.random_obstacle_pos()
        col_cylinder_id = self.p_env.createCollisionShape(self.p_env.GEOM_CYLINDER, radius=CYLINDER_R_H[0], height=CYLINDER_R_H[1])
        for i in range(0, len(self.obstacle_pos)):
            cylinder_Id = self.p_env.createMultiBody(MASS, col_cylinder_id, visualShapeId, self.obstacle_pos[i],
                                                     OBSTACLE_ORIENTATION)

        # Get robot info
        self.num_joints = self.p_env.getNumJoints(self.robotId)
        self.num_real_links = int(self.num_joints/2)
        self.num_bodies = self.p_env.getNumBodies()
        joint_info_pri = self.p_env.getJointInfo(self.robotId, 1)
        joint_info_rev = self.p_env.getJointInfo(self.robotId, 2)
        self.constraint_pri = np.array([joint_info_pri[8], joint_info_pri[9]])
        self.constraint_rev = np.array([joint_info_rev[8], joint_info_rev[9]])

    def step(self, delta_pos):
        # Actions are the changes in joints
        info = " Not Yet "
        done = False
        self.step_count += 1
        self.pos += delta_pos

        # Clip values to avoid number exceed the constraints
        self.pos[0] = self.pos[0].clip(self.constraint_pri[0], self.constraint_pri[1])
        self.pos[1:] = self.pos[1:].clip(self.constraint_rev[0], self.constraint_rev[1])

        # Set positions (action)
        motor_pos = np.insert(self.pos, 0, 0)
        self.p_env.setJointMotorControlArray(self.robotId, range(self.num_joints),
                                             controlMode=self.p_env.POSITION_CONTROL,
                                             targetPositions=motor_pos)
        for i in range(65):
            self.p_env.stepSimulation()

        # Get observation after taking action
        observation, final_pos = self.get_observation()
        observation = np.concatenate((observation, self.goal))

        # Calculate reward by the distance between end effector and goal
        distance = np.linalg.norm(final_pos - self.goal)
        reward = -distance * self.REWARD_SCALAR

        # Check if collided or not
        contact_list = []
        for i in range(0, self.num_real_links):
            index = 2 * i + 1
            contact_list.append(self.p_env.getContactPoints(self.robotId, index))
        contact = True in [True if x else False for x in contact_list]

        # Give penalty if collided
        if contact:
            self.contact_num += 1
            reward += -2 * self.REWARD_SCALAR
            self.contact = True
            self.contact_info = "   Warning!!!"

        # End the episode if maximum step or the goal is achieved
        if distance < self.DISTANT_THRESHOLD or self.step_count == self.MAX_STEP:
            done = True
            # Bonus if goal achieved
            if distance < self.DISTANT_THRESHOLD:
                reward += 2 * self.REWARD_SCALAR
                info = "is success"
                # Another bonus if no collision happened
                if not self.contact:
                    reward += 5 * self.REWARD_SCALAR

        info = info + self.contact_info

        return observation, reward, done, info

    def reset(self):
        self.contact_num = 0
        self.step_count = 0
        self.contact = False
        self.contact_info = ""
        self.goal = self.pick_goal()
        self.pos = np.zeros(self.num_joints-1)

        for i in range(0, self.num_joints):
            self.p_env.resetJointState(self.robotId, i, 0)

        reset_state, _ = self.get_observation()
        reset_state = np.concatenate((reset_state, self.goal))

        return reset_state

    def get_observation(self):
        links_global_pos = []
        links_rel_obs = []
        for i in range(self.num_real_links):
            index = 2*i + 1
            links_states = self.p_env.getLinkState(self.robotId, index)
            # add global position into observation
            links_global_pos.append(links_states[0])
            for j in range(4):
                # relative position
                # links_observation.append(links_states[0][:2] - self.obstacle_pos[j][:2])
                links_rel_obs.append(links_states[0] - self.obstacle_pos[j])

        end2goal_pos = links_states[0] - self.goal
        links_rel_obs = np.ravel(links_rel_obs)
        links_global_pos = np.ravel(links_global_pos)

        observation = np.concatenate((links_global_pos, links_rel_obs, end2goal_pos), axis=0)

        return observation, links_states[0]

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
        vis_goal_id = self.p_env.createVisualShape(self.p_env.GEOM_SPHERE, radius=SPHERE_R, rgbaColor=[1,0,0,1]) # red
        goalId = self.p_env.createMultiBody(MASS, baseVisualShapeIndex=vis_goal_id, basePosition=goal_pos)
        return goalId

    def get_joint_state(self):
        joints_state = []
        for i in range(1, self.num_joints):
            joint_state = self.p_env.getJointState(self.robotId, i)
            joint_state = joint_state[0]
            joints_state.append(joint_state)
        joints_state = np.array(joints_state)
        return joints_state

    def close(self):
        self.p_env.disconnect()
        # print('Environment close')

    @staticmethod
    def pick_goal():
        # pick a goal from defined space for each episode
        random_pos_x = np.random.uniform(8.8, 9.3)
        random_pos_y = np.random.uniform(-3.5, 3.5)
        random_pos_z = np.random.uniform(2, 4)
        goal_pos = np.array([random_pos_x, random_pos_y, random_pos_z])
        return goal_pos

    @staticmethod
    def random_obstacle_pos():
        # Obs distribution 1:
        # Randomly pick four positions and the distance between either two of them should larger than the radius
        # Goals should probably be achievable or the transitions might not be a good data for training policy

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

        # Obs distribution 2:
        # Add bias to default obstacle positions
        obstacles_pos = np.array(OBSTACLE_POS)
        for i in range(len(obstacles_pos)):
            obstacles_pos[i][0] = obstacles_pos[i][0] + np.random.uniform(-0.5, 0.5)
            obstacles_pos[i][1] = obstacles_pos[i][1] + np.random.uniform(-0.25, 0.25)

        return obstacles_pos

