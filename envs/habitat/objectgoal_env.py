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

from constants import get_habitat_labels
import random


class ObjectGoal_Env(habitat.RLEnv):
    """The Object Goal Navigation environment class. The class is responsible
    for loading the dataset, generating episodes, and computing evaluation
    metrics.
    """

    def __init__(self, args, rank, config_env, dataset):
        self.args = args
        self.rank = rank

        super().__init__(config_env, dataset)



        self.habitat_labels = get_habitat_labels(args.dataset)

        # print("=====goals_by_category", dataset.goals_by_category)
        self.config_env = config_env
        # Loading dataset info file
        self.split = config_env.DATASET.SPLIT
        self.episodes_dir = config_env.DATASET.EPISODES_DIR.format(
            split=self.split)


        # episode_total = 0

        # for i in range(len(config_env.DATASET.CONTENT_SCENES)):
        #     scene_name = config_env.DATASET.CONTENT_SCENES[i]

        #     episodes_file = self.episodes_dir + \
        #         "content/{}.json.gz".format(scene_name)

        #     print("Loading episodes from: {}".format(episodes_file))
        #     with gzip.open(episodes_file, 'r') as f:
        #         eps_data = json.loads(
        #             f.read().decode('utf-8'))["episodes"]
        #     print("============= eps_data",len(eps_data))
            # episode_total += len(eps_data)

        # print("episode_total:", episode_total)
        # exit(0)
            # self.eps_data_idx = 0
            # self.last_scene_path = self.scene_path
            # print("Changing scene: {}/{}".format(self.rank, self.scene_name))



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
        self.info['repeat'] = False

        self.scene_list = []

        self.scene_count = 0



    def load_new_episode(self):


        args = self.args
        self.scene_path = self.habitat_env.sim.config.sim_cfg.scene_id
        self.scene_name = self.scene_path.split("/")[-1].split(".")[0]
        self.episode = self.habitat_env.current_episode
        self.episode_len = len(self.habitat_env.episodes)

        goal_name = self.episode.object_category
        goal_idx = self.habitat_labels[goal_name]

        self.goal_idx = goal_idx
        self.goal_name = goal_name

        self.starting_distance = self.episode.info["geodesic_distance"]

        self.prev_distance = self.starting_distance


   

    def reset(self):
        """Resets the environment to a new episode.

        Returns:
            obs (ndarray): RGBD observations (4 x H x W)
            info (dict): contains timestep, pose, goal category and
                         evaluation metric info
        """
        args = self.args


        obs = super().reset()
        self.load_new_episode()

        new_scene = False
        if self.episode_no == self.episode_len-1:
            new_scene = True
            self.episode_no = 0

        self.episode_no += 1

        # Initializations
        self.timestep = 0
        self.stopped = False
        self.path_length = 1e-5
        self.trajectory_states = []

        if new_scene:
            self.scene_count += 1
            if self.scene_count == len(self.config_env.DATASET.CONTENT_SCENES):
                self.info['repeat'] = True
            new_scene = False


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
        self.info['scene_id'] = self.habitat_env.current_episode.scene_id
        self.info['episode_id'] = self.habitat_env.current_episode.episode_id


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
        args = self.args
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

            if self.timestep < args.max_episode_length-1:
                self.info['agent_success'] = 1
        else:
            spl, softspl, success, dist = self.get_metrics()
            self.info['distance_to_goal'] = dist
            self.info['spl'] = spl
            self.info['softspl'] = softspl
            self.info['success'] = success                
            self.info['agent_success'] = agent_success                



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

        dist = self._env.get_metrics()["distance_to_goal"]
        softspl = self._env.get_metrics()["softspl"]
        success = self._env.get_metrics()["success"]
        spl = self._env.get_metrics()["spl"]


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
        
        return dx, dy, do
