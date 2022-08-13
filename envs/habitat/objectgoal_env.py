import json
import bz2
import gzip
import _pickle as cPickle
import gym
import numpy as np
import quaternion
import skimage.morphology
import habitat

from envs.utils.fmm_planner import FMMPlanner
from constants import coco_categories
import envs.utils.pose as pu

from constants import habitat_labels

class ObjectGoal_Env(habitat.RLEnv):
    """The Object Goal Navigation environment class. The class is responsible
    for loading the dataset, generating episodes, and computing evaluation
    metrics.
    """

    def __init__(self, args, rank, config_env, dataset):
        self.args = args
        self.rank = rank

        super().__init__(config_env, dataset)

        # print("=====goals_by_category", dataset.goals_by_category)

        # Loading dataset info file
        self.split = config_env.DATASET.SPLIT
        self.episodes_dir = config_env.DATASET.EPISODES_DIR.format(
            split=self.split)

        # dataset_info_file = self.episodes_dir + \
        #     "{split}_info.pbz2".format(split=self.split)
        # with bz2.BZ2File(dataset_info_file, 'rb') as f:
        #     self.dataset_info = cPickle.load(f)

        # Specifying action and observation space
        self.action_space = gym.spaces.Discrete(3)

        self.observation_space = gym.spaces.Box(0, 255,
                                                (3, args.frame_height,
                                                 args.frame_width),
                                                dtype='uint8')

        # Initializations
        self.episode_no = 0

        # Scene info
        self.last_scene_path = None
        self.scene_path = None
        self.scene_name = None

        # Episode Dataset info
        self.eps_data = None
        self.eps_data_idx = None
        self.gt_planner = None
        self.object_boundary = None
        self.goal_idx = None
        self.goal_name = None
        self.map_obj_origin = None
        self.starting_loc = None
        self.starting_distance = None

        # Episode tracking info
        self.curr_distance = None
        self.prev_distance = None
        self.timestep = None
        self.stopped = None
        self.path_length = None
        self.last_sim_location = None
        self.trajectory_states = []
        self.info = {}
        self.info['distance_to_goal'] = None
        self.info['spl'] = None
        self.info['success'] = None
        self.info['softspl'] = None
        self.info['agent_success'] = None


    def load_new_episode(self):


        args = self.args
        self.scene_path = self.habitat_env.sim.config.sim_cfg.scene_id

        scene_name = self.scene_path.split("/")[-1].split(".")[0]

        if self.scene_path != self.last_scene_path:
            episodes_file = self.episodes_dir + \
                "content/{}.json.gz".format(scene_name)

            print("Loading episodes from: {}".format(episodes_file))
            with gzip.open(episodes_file, 'r') as f:
                self.eps_data = json.loads(
                    f.read().decode('utf-8'))["episodes"]

            self.eps_data_idx = 0
            self.last_scene_path = self.scene_path

        # Load episode info
        episode = self.eps_data[self.eps_data_idx]

        print("episode============", episode)

        self.eps_data_idx += 1
        self.eps_data_idx = self.eps_data_idx % len(self.eps_data)
        pos = episode["start_position"]
        rot = episode["start_rotation"]
        # rot = quaternion.from_float_array(episode["start_rotation"])

        # print("pos", pos)
        # print("rot", rot)

        goal_name = episode["object_category"]
        # goal_idx = episode["object_id"]
        goal_idx = habitat_labels[goal_name]
        # floor_idx = episode["floor_id"]

        self.goal_idx = goal_idx
        self.goal_name = goal_name
        # self.map_obj_origin = map_obj_origin

        # self.starting_distance = self.gt_planner.fmm_dist[self.starting_loc]\
        #     / 20.0 + self.object_boundary
        self.starting_distance = episode["info"]["geodesic_distance"]

        self.prev_distance = self.starting_distance


   

    def reset(self):
        """Resets the environment to a new episode.

        Returns:
            obs (ndarray): RGBD observations (4 x H x W)
            info (dict): contains timestep, pose, goal category and
                         evaluation metric info
        """
        args = self.args
        new_scene = self.episode_no % args.num_train_episodes == 0

        self.episode_no += 1

        # Initializations
        self.timestep = 0
        self.stopped = False
        self.path_length = 1e-5
        self.trajectory_states = []

        # print(">>>>>>>>>", self.habitat_env.sim.config)


        obs = super().reset()
        # print("===========obs", obs)
        print("================= scene_id", self.habitat_env.sim.config.sim_cfg.scene_id)
        if new_scene:
            # self.scene_name = self.habitat_env.sim.config.SCENE
            self.scene_name = self.habitat_env.sim.config.sim_cfg.scene_id,

            print("Changing scene: {}/{}".format(self.rank, self.scene_name))

        self.scene_path = self.habitat_env.sim.config.sim_cfg.scene_id

        # if self.split == "val" or self.split == "val_mini":
        #     obs = self.load_new_episode()
        # else:
        #     obs = self.generate_new_episode()

        self.load_new_episode()


        rgb = obs['rgb'].astype(np.uint8)
        depth = obs['depth']
        state = np.concatenate((rgb, depth), axis=2).transpose(2, 0, 1)
        self.last_sim_location = self.get_sim_location()

        # Set info
        self.info['time'] = self.timestep
        self.info['sensor_pose'] = [0., 0., 0.]
        self.info['current_pose'] = self.get_sim_location()
        self.info['goal_cat_id'] = self.goal_idx
        self.info['goal_name'] = self.goal_name
        self.info['scene_id'] = self.habitat_env.sim.config.sim_cfg.scene_id
        # self.info['episode_idx'] = self. 

        return state, self.info

    def step(self, action):
        """Function to take an action in the environment.

        Args:
            action (dict):
                dict with following keys:
                    'action' (int): 0: stop, 1: forward, 2: left, 3: right

        Returns:
            obs (ndarray): RGBD observations (4 x H x W)
            reward (float): amount of reward returned after previous action
            done (bool): whether the episode has ended
            info (dict): contains timestep, pose, goal category and
                         evaluation metric info
        """
        action = action["action"]
        if action == 0:
            self.stopped = True
            # Not sending stop to simulator, resetting manually
            # action = 3

        obs, rew, done, _ = super().step(action)

        # Get pose change
        dx, dy, do = self.get_pose_change()
        self.info['sensor_pose'] = [dx, dy, do]
        self.info['current_pose'] = self.get_sim_location()
        # print("current pose", self.info['current_pose'])


        self.path_length += pu.get_l2_distance(0, dx, 0, dy)

        softspl, spl, success, agent_success, dist = 0., 0., 0., 0., 0.
        if done:
            spl, softspl, success, dist = self.get_metrics()
            self.info['distance_to_goal'] = dist
            self.info['spl'] = spl
            self.info['softspl'] = softspl
            self.info['success'] = success

            if self.timestep<490:
                self.info['agent_success'] = 1
            # print("dist", dist)
            # print("spl", spl)
            # print("success", success)
        else:
            spl, softspl, success, dist = self.get_metrics()
            self.info['distance_to_goal'] = dist
            self.info['spl'] = spl
            self.info['softspl'] = softspl
            self.info['success'] = success                
            self.info['agent_success'] = agent_success                

            # print("success!!!")
            # exit(0)


        rgb = obs['rgb'].astype(np.uint8)
        depth = obs['depth']
        state = np.concatenate((rgb, depth), axis=2).transpose(2, 0, 1)

        self.timestep += 1
        self.info['time'] = self.timestep

        return state, rew, done, self.info

    def get_reward_range(self):
        """This function is not used, Habitat-RLEnv requires this function"""
        return (0., 1.0)

    def get_reward(self, observations):

        reward = -0.0001   # slack reward = -10e-4
        if self._env.get_metrics()["success"]:
            reward += 2.5 # success reward 

        return reward

    def get_metrics(self):
        """This function computes evaluation metrics for the Object Goal task

        Returns:
            spl (float): Success weighted by Path Length
                        (See https://arxiv.org/pdf/1807.06757.pdf)
            success (int): 0: Failure, 1: Successful
            dist (float): Distance to Success (DTS),  distance of the agent
                        from the success threshold boundary in meters.
                        (See https://arxiv.org/pdf/2007.00643.pdf)
        """
        # curr_loc = self.sim_continuous_to_sim_map(self.get_sim_location())




        # print("goals", self.episode)



        # print("metrics",self._env.get_metrics())
        dist = self._env.get_metrics()["distance_to_goal"]
        softspl = self._env.get_metrics()["softspl"]
        success = self._env.get_metrics()["success"]
        spl = self._env.get_metrics()["spl"]

        # print（）
        # print("success ",success)

        # print("spl ",spl)
        # if dist < 0.2:
        #     success = 1
        # else:
        #     success = 0
        # spl = min(success * self.starting_distance / self.path_length, 1)
        return spl, softspl, success, dist

    def get_done(self, observations):

        if self.info['time'] >= self.args.max_episode_length - 1:
            done = True
        elif self.stopped:
            done = True
        else:
            done = False
        return done

    def get_info(self, observations):
        """This function is not used, Habitat-RLEnv requires this function"""
        info = {}
        return info

    def get_spaces(self):
        """Returns observation and action spaces for the ObjectGoal task."""
        return self.observation_space, self.action_space

    def get_sim_location(self):
        """Returns x, y, o pose of the agent in the Habitat simulator."""

        agent_state = super().habitat_env.sim.get_agent_state(0)
        x = -agent_state.position[2]
        y = -agent_state.position[0]
        axis = quaternion.as_euler_angles(agent_state.rotation)[0]
        if (axis % (2 * np.pi)) < 0.1 or (axis %
                                          (2 * np.pi)) > 2 * np.pi - 0.1:
            o = quaternion.as_euler_angles(agent_state.rotation)[1]
        else:
            o = 2 * np.pi - quaternion.as_euler_angles(agent_state.rotation)[1]
        if o > np.pi:
            o -= 2 * np.pi
        return x, y, o

    def get_pose_change(self):
        """Returns dx, dy, do pose change of the agent relative to the last
        timestep."""
        curr_sim_pose = self.get_sim_location()
        dx, dy, do = pu.get_rel_pose_change(
            curr_sim_pose, self.last_sim_location)
        self.last_sim_location = curr_sim_pose
        # print("dx dy do", dx, dy ,do)
        return dx, dy, do
