from cgi import print_directory
import tensorflow
from collections import deque, defaultdict
import os
import logging
import time
import json
import gym
import torch.nn as nn
import torch
import numpy as np
from datetime import datetime

from model import Semantic_Mapping, RL_Identify_Policy, RL_Explore_Policy
from utils.storage import Identify_GlobalRolloutStorage, Explore_GlobalRolloutStorage
from utils.log_writter import log_writter
from envs import make_vec_envs
from arguments import get_args
import algo

import cv2

import matplotlib.pyplot as plt

from GLtree.interval_tree import RedBlackTree, Node, BLACK, RED, NIL
from GLtree.octree import GL_tree

os.environ["OMP_NUM_THREADS"] = "1"


def main():
    args = get_args()
    # print(args.load_2d)
    # exit(0)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    args.exp_name = args.exp_name +"-"+ datetime.now().strftime("%d-%m-%H:%M:%S")

    # Setup Logging
    log_dir = "{}/models/{}/".format(args.dump_location, args.exp_name)
    dump_dir = "{}/dump/{}/".format(args.dump_location, args.exp_name)
    tb_dir = "{}/tb/{}/".format(args.dump_location, args.exp_name)

    log_wr = log_writter("{}/tb/{}/".format(args.dump_location, args.exp_name))


    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)
    if not os.path.exists(tb_dir):
        os.makedirs(tb_dir)


    logging.basicConfig(
        filename=log_dir + 'train.log',
        level=logging.INFO)
    print("Dumping at {}".format(log_dir))
    print(args)
    logging.info(args)

    # Logging and loss variables
    num_scenes = args.num_processes

    # print(num_scenes)
    # exit(0)

    num_episodes = int(args.num_eval_episodes)

    device = args.device = torch.device(args.policy_gpu_id if args.cuda else "cpu")


    g_masks = torch.ones(num_scenes).float().to(device)

    best_g_reward = -np.inf

    if args.eval:
        episode_success = []
        episode_spl = []
        episode_dist = []
        episode_softspl = []
        episode_agent_success = []
        for _ in range(args.num_processes):
            episode_softspl.append(deque(maxlen=num_episodes))
            episode_agent_success.append(deque(maxlen=num_episodes))
            episode_success.append(deque(maxlen=num_episodes))
            episode_spl.append(deque(maxlen=num_episodes))
            episode_dist.append(deque(maxlen=num_episodes))

    else:
        episode_success = deque(maxlen=1000)
        episode_spl = deque(maxlen=1000)
        episode_dist = deque(maxlen=1000)
        episode_softspl = deque(maxlen=1000)
        episode_agent_success = deque(maxlen=1000)


    cat_pred_threshold = []

    for _ in range(args.num_sem_categories-1):
        cat_pred_threshold.append([])

    timestep_threshold = np.zeros(500) 
    timestep_count_threshold = np.zeros(500) + 1e-5




    finished = np.zeros((args.num_processes))
    wait_env = np.zeros((args.num_processes))

    g_episode_rewards = deque(maxlen=1000)

    explore_value_losses = deque(maxlen=1000)
    explore_action_losses = deque(maxlen=1000)
    explore_dist_entropies = deque(maxlen=1000)


    identify_value_losses = deque(maxlen=1000)
    identify_action_losses = deque(maxlen=1000)
    identify_dist_entropies = deque(maxlen=1000)

    per_step_g_rewards = deque(maxlen=1000)

    per_step_explore_rewards = deque(maxlen=1000)
    per_step_env_rewards = deque(maxlen=1000)




    g_process_rewards = np.zeros((num_scenes))

    # Starting environments
    torch.set_num_threads(1)
    envs = make_vec_envs(args)
    # exit(0)
    obs, infos = envs.reset()

    torch.set_grad_enabled(False)

    gl_tree_list = []
    for e in range(num_scenes):
        gl_tree_list.append(GL_tree(args))

    # Initialize map variables:.
    # Full map consists of multiple channels containing the following:
    # 1. Obstacle Map
    # 2. Exploread Area
    # 3. Current Agent Location
    # 4. Past Agent Locations
    # 5,6,7,.. : Semantic Categories
    nc = args.num_sem_categories + 4  # num channels


    # Calculating full and local map sizes
    map_size = args.map_size_cm // args.map_resolution
    full_w, full_h = map_size, map_size
    local_w = int(full_w / args.global_downscaling)
    local_h = int(full_h / args.global_downscaling)

    # 8 global goal selection
    global_action_selection_list = np.asarray([[0,0],[0, int(local_w/2)],[0, local_w-1],\
                    [int(local_h/2), 0], [int(local_w/2), local_w-1],\
                    [int(local_h-1), 0], [int(local_h-1), int(local_w/2)], [int(local_h-1), local_w-1]])


    # Initializing full and local map
    # points_channel_num = 12
    points_channel_num = 3 + args.num_sem_categories + 1 + 1
    full_map = torch.zeros(num_scenes, nc, full_w, full_h).float().to(device)
    local_map = torch.zeros(num_scenes, nc, local_w,
                            local_h).float().to(device)


    # points_channel_num_3d = 3 + 1 + 1
    observation_points = torch.zeros(num_scenes, points_channel_num, args.map_point_size)

    # observation_points_3d = torch.zeros(num_scenes, points_channel_num_3d, args.map_point_size)


    print("=====full map=====", full_map.shape)

    # Initial full and local pose
    full_pose = torch.zeros(num_scenes, 3).float().to(device)
    local_pose = torch.zeros(num_scenes, 3).float().to(device)

    # Origin of local map
    origins = np.zeros((num_scenes, 3))

    # Local Map Boundaries
    lmb = np.zeros((num_scenes, 4)).astype(int)

    # Planner pose inputs has 7 dimensions
    # 1-3 store continuous global agent location
    # 4-7 store local map boundaries
    planner_pose_inputs = np.zeros((num_scenes, 7))

    def get_local_map_boundaries(agent_loc, local_sizes, full_sizes):
        loc_r, loc_c = agent_loc
        local_w, local_h = local_sizes
        full_w, full_h = full_sizes

        if args.global_downscaling > 1:
            gx1, gy1 = loc_r - local_w // 2, loc_c - local_h // 2
            gx2, gy2 = gx1 + local_w, gy1 + local_h
            if gx1 < 0:
                gx1, gx2 = 0, local_w
            if gx2 > full_w:
                gx1, gx2 = full_w - local_w, full_w

            if gy1 < 0:
                gy1, gy2 = 0, local_h
            if gy2 > full_h:
                gy1, gy2 = full_h - local_h, full_h
        else:
            gx1, gx2, gy1, gy2 = 0, full_w, 0, full_h

        return [gx1, gx2, gy1, gy2]

    def init_map_and_pose():
        full_map.fill_(0.)
        observation_points.fill_(0.)
        full_pose.fill_(0.)
        full_pose[:, :2] = args.map_size_cm / 100.0 / 2.0

        locs = full_pose.cpu().numpy()
        planner_pose_inputs[:, :3] = locs
        for e in range(num_scenes):
            r, c = locs[e, 1], locs[e, 0]
            loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                            int(c * 100.0 / args.map_resolution)]

            full_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0

            lmb[e] = get_local_map_boundaries((loc_r, loc_c),
                                              (local_w, local_h),
                                              (full_w, full_h))

            planner_pose_inputs[e, 3:] = lmb[e]
            origins[e] = [lmb[e][2] * args.map_resolution / 100.0,
                          lmb[e][0] * args.map_resolution / 100.0, 0.]

        for e in range(num_scenes):
            local_map[e] = full_map[e, :,
                                    lmb[e, 0]:lmb[e, 1],
                                    lmb[e, 2]:lmb[e, 3]]
            local_pose[e] = full_pose[e] - \
                torch.from_numpy(origins[e]).to(device).float()

        for e in range(num_scenes):
            gl_tree_list[e].reset_gltree()


       
    def init_map_and_pose_for_env(e):
        full_map[e].fill_(0.)
        observation_points[e].fill_(0.)
        full_pose[e].fill_(0.)
        full_pose[e, :2] = args.map_size_cm / 100.0 / 2.0
        gl_tree_list[e].reset_gltree()


        locs = full_pose[e].cpu().numpy()
        planner_pose_inputs[e, :3] = locs
        r, c = locs[1], locs[0]
        loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                        int(c * 100.0 / args.map_resolution)]

        full_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0

        lmb[e] = get_local_map_boundaries((loc_r, loc_c),
                                          (local_w, local_h),
                                          (full_w, full_h))

        planner_pose_inputs[e, 3:] = lmb[e]
        origins[e] = [lmb[e][2] * args.map_resolution / 100.0,
                      lmb[e][0] * args.map_resolution / 100.0, 0.]

        local_map[e] = full_map[e, :, lmb[e, 0]:lmb[e, 1], lmb[e, 2]:lmb[e, 3]]
        local_pose[e] = full_pose[e] - \
            torch.from_numpy(origins[e]).to(device).float()


    def update_intrinsic_rew(e):
        prev_explored_area = full_map[e, 1].sum(1).sum(0)
        full_map[e, :, lmb[e, 0]:lmb[e, 1], lmb[e, 2]:lmb[e, 3]] = \
            local_map[e]
        curr_explored_area = full_map[e, 1].sum(1).sum(0)
        intrinsic_rews[e] = curr_explored_area - prev_explored_area
        intrinsic_rews[e] *= (args.map_resolution / 100.)**2  # to m^2

    init_map_and_pose()


    # Global policy observation space
    ngc = 8 + args.num_sem_categories
    es = 3
    map_observation_space = gym.spaces.Box(0, 1,
                                         (ngc,
                                          local_w,
                                          local_h), dtype='uint8')

    points_observation_space = gym.spaces.Box(0, 1,
                                         (points_channel_num, 
                                          args.map_point_size), dtype='float32')  





    explore_action_space = gym.spaces.Discrete(8)

    identify_action_space = gym.spaces.Discrete(11)



    # Global policy recurrent layer size
    hidden_size = args.global_hidden_size

    # Semantic Mapping
    sem_map_module = Semantic_Mapping(args).to(device)
    sem_map_module.eval()

    explore_policy = RL_Explore_Policy(map_observation_space.shape, points_observation_space.shape, explore_action_space,
                         model_type=1,
                         base_kwargs={'recurrent': args.use_recurrent_global,
                                      'hidden_size': hidden_size,
                                      'num_sem_categories': ngc - 8
                                      }).to(device)

    explore_agent = algo.PPO(explore_policy, args.clip_param, args.ppo_epoch,
                       args.num_mini_batch, args.value_loss_coef,
                       args.entropy_coef, lr=args.lr, eps=args.eps,
                       max_grad_norm=args.max_grad_norm)

    identify_policy = RL_Identify_Policy(points_observation_space.shape, identify_action_space,
                                model_type=1,
                                base_kwargs={
                                            'hidden_size': hidden_size,
                                            'num_sem_categories': ngc - 8,
                                            'points_channel_num': points_channel_num
                                            }).to(device)

    identify_agent = algo.PPO_3d(identify_policy, args.clip_param, args.ppo_epoch,
                       args.num_mini_batch, args.value_loss_coef,
                       args.entropy_coef, lr=args.lr, eps=args.eps,
                       max_grad_norm=args.max_grad_norm)

    global_input = torch.zeros(num_scenes, ngc, local_w, local_h)
    global_orientation = torch.zeros(num_scenes, 1).long()
    intrinsic_rews = torch.zeros(num_scenes).to(device)
    extras = torch.zeros(num_scenes, 3)

    # Storage
    explore_rollouts   = Explore_GlobalRolloutStorage(args.num_global_steps,
                                      num_scenes, map_observation_space.shape, points_observation_space.shape, 
                                      explore_action_space, explore_policy.rec_state_size,
                                      es).to(device)

    identify_rollouts = Identify_GlobalRolloutStorage(args.num_global_steps,
                                      num_scenes, points_observation_space.shape, 
                                      identify_action_space, identify_policy.rec_state_size,
                                      es).to(device)




    if args.load_explore != "0" and args.load_3d != "0":
        print("Loading exploration policy {}".format(args.load_explore))
        print("Loading identify policy {}".format(args.load_identify))

        explore_state_dict = torch.load(args.load_explore,
                                map_location=lambda storage, loc: storage)
        explore_policy.load_state_dict(explore_state_dict)

        identify_state_dict = torch.load(args.load_identify,
                                map_location=lambda storage, loc: storage)
        identify_policy.load_state_dict(identify_state_dict)

    if args.eval:
        explore_policy.eval()
        identify_policy.eval()

    # Predict semantic map from frame 1
    poses = torch.from_numpy(np.asarray(
        [infos[env_idx]['sensor_pose'] for env_idx in range(num_scenes)])
    ).float().to(device)

    #global pose in world coordinate
    world_poses = torch.from_numpy(np.asarray(
        [infos[env_idx]['current_pose'] for env_idx in range(num_scenes)])
    ).float().to(device)


    goal_cat_id = torch.from_numpy(np.asarray(
        [infos[env_idx]['goal_cat_id'] for env_idx
         in range(num_scenes)]))


    timestep_array = torch.from_numpy(np.asarray(
        [0*1.0/500 for env_idx
            in range(num_scenes)]))



    _, local_map, _, local_pose, observation_points= \
        sem_map_module(obs, poses, local_map, local_pose, origins, observation_points, goal_cat_id, gl_tree_list, infos, wait_env, args)



    # Compute Global policy input
    locs = local_pose.cpu().numpy()
    global_input = torch.zeros(num_scenes, ngc, local_w, local_h)
    global_orientation = torch.zeros(num_scenes, 1).long()

    for e in range(num_scenes):
        r, c = locs[e, 1], locs[e, 0]
        loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                        int(c * 100.0 / args.map_resolution)]

        local_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.
        global_orientation[e] = int((locs[e, 2] + 180.0) / 5.)

    global_input[:, 0:4, :, :] = local_map[:, 0:4, :, :].detach()
    global_input[:, 4:8, :, :] = nn.MaxPool2d(args.global_downscaling)(
        full_map[:, 0:4, :, :])
    global_input[:, 8:, :, :] = local_map[:, 4:, :, :].detach()




    extras = torch.zeros(num_scenes, 3)
    extras[:, 0] = global_orientation[:, 0]
    extras[:, 1] = goal_cat_id
    extras[:, 2] = timestep_array   #to the finish

    explore_rollouts.obs_map[0].copy_(global_input)   
    explore_rollouts.obs_points[0].copy_(observation_points)
    explore_rollouts.extras[0].copy_(extras)


    identify_rollouts.obs_points[0].copy_(observation_points)
    identify_rollouts.extras[0].copy_(extras)

    # Run Global Policy (global_goals = Long-Term Goal)
    explore_value, explore_action, explore_action_log_prob, explore_rec_states = \
        explore_policy.act(
            explore_rollouts.obs_map[0],
            explore_rollouts.obs_points[0],
            explore_rollouts.rec_states[0],
            explore_rollouts.masks[0],
            extras=explore_rollouts.extras[0],
            deterministic=False
        )

    # Run Global Policy (global_goals = Long-Term Goal)
    identify_value, identify_action, identify_action_log_prob, identify_rec_states = \
        identify_policy.act(
            identify_rollouts.obs_points[0],
            identify_rollouts.masks[0],
            extras=identify_rollouts.extras[0],
            deterministic=False
        )

    cpu_actions = explore_action.cpu().numpy()

    global_goals = [global_action_selection_list[cpu_actions[action]]
                    for action in range(num_scenes)]

    
    confidence_thres = args.sem_pred_lower_bound + identify_action.cpu().numpy()/10 * (1 - args.sem_pred_lower_bound)




    goal_maps = [np.zeros((local_w, local_h)) for _ in range(num_scenes)]

    for e in range(num_scenes):
        goal_maps[e][global_goals[e][1], global_goals[e][0]] = 1

    planner_inputs = [{} for e in range(num_scenes)]
    for e, p_input in enumerate(planner_inputs):
        p_input['map_pred'] = local_map[e, 0, :, :].cpu().numpy()
        p_input['exp_pred'] = local_map[e, 1, :, :].cpu().numpy()
        p_input['pose_pred'] = planner_pose_inputs[e]
        p_input['goal'] = goal_maps[e]  # global_goals[e]
        p_input['new_goal'] = 1
        p_input['found_goal'] = 0
        p_input['wait'] = wait_env[e] or finished[e]
        if args.visualize or args.print_images:
            local_map[e, -1, :, :] = 1e-5
            p_input['sem_map_pred'] = local_map[e, 4:, :, :
                                                ].argmax(0).cpu().numpy()

    # import time
    # t_s = time.time()
    obs, _, done, infos = envs.plan_act_and_preprocess(planner_inputs)
    # print("sem 00", time.time() - t_s)

    # print(infos)
    start = time.time()
    g_reward = 0

    torch.set_grad_enabled(False)
    spl_per_category = defaultdict(list)
    success_per_category = defaultdict(list)




    for step in range(args.num_training_frames // args.num_processes + 1):
        if finished.sum() == args.num_processes:
            break

        if step%1000 == 0 :
            with open('{}/{}_spl_per_cat_pred_thr_{}.json'.format(
                dump_dir, args.split, str(step)), 'w') as f:
                json.dump(spl_per_category, f)
            with open('{}/{}_success_per_cat_pred_thr_{}.json'.format(
                dump_dir, args.split, str(step)), 'w') as f:
                json.dump(success_per_category, f)

        g_step = (step // args.num_local_steps) % args.num_global_steps
        l_step = step % args.num_local_steps

        # ------------------------------------------------------------------
        # Reinitialize variables when episode ends
        l_masks = torch.FloatTensor([0 if x else 1
                                     for x in done]).to(device)
        g_masks *= l_masks

        for e, x in enumerate(done):
            if x:

                spl = infos[e]['spl']
                success = infos[e]['success']
                dist = infos[e]['distance_to_goal']

                agent_success = infos[e]['agent_success']
                softspl = infos[e]['softspl']

                if args.eval:

                    episode_success[e].append(success)
                    episode_spl[e].append(spl)
                    episode_dist[e].append(dist)
                    episode_agent_success[e].append(agent_success)
                    episode_softspl[e].append(softspl)

                    if infos[e]["repeat"]:
                        finished[e] = 1
                else:
                    episode_success.append(success)
                    episode_spl.append(spl)
                    episode_dist.append(dist)
                    episode_agent_success.append(agent_success)
                    episode_softspl.append(softspl)


                spl_per_category[infos[e]['goal_name']].append(spl)
                success_per_category[infos[e]['goal_name']].append(success)

                wait_env[e] = 1.
                update_intrinsic_rew(e)
                init_map_and_pose_for_env(e)


        poses = torch.from_numpy(np.asarray(
            [infos[env_idx]['sensor_pose'] for env_idx
             in range(num_scenes)])
        ).float().to(device)

        world_poses = torch.from_numpy(np.asarray(
            [infos[env_idx]['current_pose'] for env_idx in range(num_scenes)])
        ).float().to(device)


        goal_cat_id = torch.from_numpy(np.asarray(
                [infos[env_idx]['goal_cat_id'] for env_idx
                 in range(num_scenes)]))

        _, local_map, _, local_pose, observation_points= \
            sem_map_module(obs, poses, local_map, local_pose, origins, observation_points, goal_cat_id, gl_tree_list, infos, wait_env, args)


        locs = local_pose.cpu().numpy()
        planner_pose_inputs[:, :3] = locs + origins
        local_map[:, 2, :, :].fill_(0.)  # Resetting current location channel
        for e in range(num_scenes):
            r, c = locs[e, 1], locs[e, 0]
            loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                            int(c * 100.0 / args.map_resolution)]
            local_map[e, 2:4, loc_r - 2:loc_r + 3, loc_c - 2:loc_c + 3] = 1.

        if l_step == args.num_local_steps - 1:
            for e in range(num_scenes):
                if wait_env[e] == 1:  # New episode
                    wait_env[e] = 0.
                else:
                    update_intrinsic_rew(e)

                full_map[e, :, lmb[e, 0]:lmb[e, 1], lmb[e, 2]:lmb[e, 3]] = \
                    local_map[e]
                full_pose[e] = local_pose[e] + \
                    torch.from_numpy(origins[e]).to(device).float()

                locs = full_pose[e].cpu().numpy()
                r, c = locs[1], locs[0]
                loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                                int(c * 100.0 / args.map_resolution)]

                lmb[e] = get_local_map_boundaries((loc_r, loc_c),
                                                  (local_w, local_h),
                                                  (full_w, full_h))

                planner_pose_inputs[e, 3:] = lmb[e]
                origins[e] = [lmb[e][2] * args.map_resolution / 100.0,
                              lmb[e][0] * args.map_resolution / 100.0, 0.]

                local_map[e] = full_map[e, :,
                                        lmb[e, 0]:lmb[e, 1],
                                        lmb[e, 2]:lmb[e, 3]]
                local_pose[e] = full_pose[e] - \
                    torch.from_numpy(origins[e]).to(device).float()

            locs = local_pose.cpu().numpy()
            for e in range(num_scenes):
                global_orientation[e] = int((locs[e, 2] + 180.0) / 5.)
            global_input[:, 0:4, :, :] = local_map[:, 0:4, :, :]
            global_input[:, 4:8, :, :] = \
                nn.MaxPool2d(args.global_downscaling)(
                    full_map[:, 0:4, :, :])
            global_input[:, 8:, :, :] = local_map[:, 4:, :, :].detach()
            goal_cat_id = torch.from_numpy(np.asarray(
                [infos[env_idx]['goal_cat_id'] for env_idx
                 in range(num_scenes)]))

            timestep_array = torch.from_numpy(np.asarray(
                [infos[env_idx]['timestep']*1.0/500 for env_idx
                 in range(num_scenes)]))

            extras[:, 0] = global_orientation[:, 0]
            extras[:, 1] = goal_cat_id
            extras[:, 2] = timestep_array

            # Get exploration reward and metrics
            g_reward = torch.from_numpy(np.asarray(
                [infos[env_idx]['g_reward'] for env_idx in range(num_scenes)])
            ).float().to(device)

            per_step_env_rewards.append(np.mean(g_reward.cpu().numpy()))
            per_step_explore_rewards.append(np.mean(args.intrinsic_rew_coeff * intrinsic_rews.detach().cpu().numpy()))

            g_reward += args.intrinsic_rew_coeff * intrinsic_rews.detach()

            g_process_rewards += g_reward.cpu().numpy()



            g_total_rewards = g_process_rewards * \
                (1 - g_masks.cpu().numpy())
            g_process_rewards *= g_masks.cpu().numpy()
            per_step_g_rewards.append(np.mean(g_reward.cpu().numpy()))

            if np.sum(g_total_rewards) != 0:
                for total_rew in g_total_rewards:
                    if total_rew != 0:
                        g_episode_rewards.append(total_rew)

            if step == 0:
                explore_rollouts.obs_map[0].copy_(global_input)
                explore_rollouts.obs_points[0].copy_(observation_points)
                explore_rollouts.extras[0].copy_(extras)

                identify_rollouts.obs_points[0].copy_(observation_points)
                identify_rollouts.extras[0].copy_(extras)

            else: # share the same reward
                explore_rollouts.insert(
                    global_input, observation_points, explore_rec_states,
                    explore_action, explore_action_log_prob, explore_value,
                    g_reward, g_masks, extras
                )

                identify_rollouts.insert(
                    observation_points, identify_rec_states,
                    identify_action, identify_action_log_prob, identify_value,
                    g_reward, g_masks, extras
                )


            # Sample long-term goal from global policy
            explore_value, explore_action, explore_action_log_prob, explore_rec_states = \
                explore_policy.act(
                    explore_rollouts.obs_map[g_step + 1],
                    explore_rollouts.obs_points[g_step + 1],
                    explore_rollouts.rec_states[g_step + 1],
                    explore_rollouts.masks[g_step + 1],
                    extras=explore_rollouts.extras[g_step + 1],
                    deterministic=False
                )

            identify_value, identify_action, identify_action_log_prob, _ = \
                identify_policy.act(
                    identify_rollouts.obs_points[g_step + 1],
                    identify_rollouts.masks[g_step + 1],
                    extras=identify_rollouts.extras[g_step + 1],
                    deterministic=False
                )


            cpu_actions = explore_action.cpu().numpy()

            global_goals = [global_action_selection_list[cpu_actions[action]]
                            for action in range(num_scenes)]

            global_goals = [[min(x, int(local_w - 1)),
                             min(y, int(local_h - 1))]
                            for x, y in global_goals]

            g_reward = 0
            g_masks = torch.ones(num_scenes).float().to(device)


        # Update long-term goal if target object is found
        found_goal = [0 for _ in range(num_scenes)]
        goal_maps = [np.zeros((local_w, local_h)) for _ in range(num_scenes)]

        confidence_thres = args.sem_pred_lower_bound + (identify_action.cpu().numpy()-5)/5 * (1 - args.sem_pred_lower_bound)


        for e in range(num_scenes):

            cat_pred_threshold[infos[e]['goal_cat_id']].append(confidence_thres[e])
            timestep_threshold[infos[e]['timestep']] += confidence_thres[e]
            timestep_count_threshold[infos[e]['timestep']]+=1

            sample_points_tensor = gl_tree_list[e].find_object_goal_points(gl_tree_list[e].observation_window, goal_cat_id[e], confidence_thres[e])

            if sample_points_tensor is not None:
                sample_points_tensor[:,:2] = sample_points_tensor[:,:2] - origins[e, :2] * 100
                sample_points_tensor[:,:3] = sample_points_tensor[:,:3] / args.map_resolution 
                sample_points_tensor = sample_points_tensor.astype(np.int32)
                sample_points_tensor = sample_points_tensor[:,:2]

                sample_points_tensor = sample_points_tensor[np.where((sample_points_tensor[:, 0]>=0) & (sample_points_tensor[:, 0]<local_w) & (sample_points_tensor[:, 1]>=0) & (sample_points_tensor[:, 1]<local_h))]


            if sample_points_tensor is not None and sample_points_tensor.shape[0]>0:
                goal_maps[e][sample_points_tensor[:,1], sample_points_tensor[:,0]] = 1

                found_goal[e] = 1
            else:
                goal_maps[e][global_goals[e][1], global_goals[e][0]] = 1




        # ------------------------------------------------------------------


        # ------------------------------------------------------------------
        # Take action and get next observation
        planner_inputs = [{} for e in range(num_scenes)]
        for e, p_input in enumerate(planner_inputs):
            p_input['map_pred'] = local_map[e, 0, :, :].cpu().numpy()
            p_input['exp_pred'] = local_map[e, 1, :, :].cpu().numpy()
            p_input['pose_pred'] = planner_pose_inputs[e]
            p_input['goal'] = goal_maps[e]  # global_goals[e]
            p_input['new_goal'] = l_step == args.num_local_steps - 1
            p_input['found_goal'] = found_goal[e]
            p_input['wait'] = wait_env[e] or finished[e]
            if args.visualize or args.print_images:

                local_map_thres = local_map[e, 4:, :, :]
                local_map_thres[-1, :, :] = 1e-5
                p_input['sem_map_pred'] = local_map_thres.argmax(0).cpu().numpy()


        obs, _, done, infos = envs.plan_act_and_preprocess(planner_inputs)


        # Training
        torch.set_grad_enabled(True)
        if g_step % args.num_global_steps == args.num_global_steps - 1 \
                and l_step == args.num_local_steps - 1:
            if not args.eval:
                explore_next_value = explore_policy.get_value(
                    explore_rollouts.obs_map[-1],
                    explore_rollouts.obs_points[-1],
                    explore_rollouts.rec_states[-1],
                    explore_rollouts.masks[-1],
                    extras=explore_rollouts.extras[-1]
                ).detach()

                explore_rollouts.compute_returns(explore_next_value, args.use_gae,
                                           args.gamma, args.tau)
                explore_value_loss, explore_action_loss, explore_dist_entropy = \
                    explore_agent.update(explore_rollouts)
                explore_value_losses.append(explore_value_loss)
                explore_action_losses.append(explore_action_loss)
                explore_dist_entropies.append(explore_dist_entropy)


                identify_next_value = identify_policy.get_value(
                    identify_rollouts.obs_points[-1],
                    extras=identify_rollouts.extras[-1]       
                    ).detach()

                identify_rollouts.compute_returns(identify_next_value, args.use_gae,
                                           args.gamma, args.tau)
                identify_value_loss, identify_action_loss, identify_dist_entropy = \
                    identify_agent.update(identify_rollouts)
                identify_value_losses.append(identify_value_loss)
                identify_action_losses.append(identify_action_loss)
                identify_dist_entropies.append(identify_dist_entropy)


            explore_rollouts.after_update()
            identify_rollouts.after_update()

        torch.set_grad_enabled(False)
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Logging
        if step % args.log_interval == 0:
            end = time.time()
            time_elapsed = time.gmtime(end - start)
            log = " ".join([
                "Time: {0:0=2d}d".format(time_elapsed.tm_mday - 1),
                "{},".format(time.strftime("%Hh %Mm %Ss", time_elapsed)),
                "num timesteps {},".format(step * num_scenes),
                "FPS {},".format(int(step * num_scenes / (end - start)))
            ])

            log += "\n\tRewards:"

            # each category threshold
            for cat_id in range(args.num_sem_categories-1):
                if len(cat_pred_threshold[cat_id])==0:
                    continue
                log_wr.writer.add_scalar("train/category/"+str(cat_id), np.mean(cat_pred_threshold[cat_id]), step)

 

            if len(g_episode_rewards) > 0:

                log_wr.writer.add_scalar("train/step/reward/mean", np.mean(per_step_g_rewards) , step)
                log_wr.writer.add_scalar("train/step/reward/median", np.median(per_step_g_rewards) , step)

                log_wr.writer.add_scalar("train/step/explore_reward/mean", np.mean(per_step_explore_rewards) , step)
                log_wr.writer.add_scalar("train/step/explore_reward/median", np.median(per_step_explore_rewards) , step)

                log_wr.writer.add_scalar("train/step/env_reward/mean", np.mean(per_step_env_rewards) , step)
                log_wr.writer.add_scalar("train/step/env_reward/median", np.median(per_step_env_rewards) , step)


                log_wr.writer.add_scalar("train/eps/reward/mean", np.mean(g_episode_rewards) , step)
                log_wr.writer.add_scalar("train/eps/reward/median", np.median(g_episode_rewards) , step)
                log_wr.writer.add_scalar("train/eps/reward/min", np.min(g_episode_rewards) , step)
                log_wr.writer.add_scalar("train/eps/reward/max", np.max(g_episode_rewards) , step)

                log += " ".join([
                    " Global step mean/med rew:",
                    "{:.4f}/{:.4f},".format(
                        np.mean(per_step_g_rewards),
                        np.median(per_step_g_rewards)),
                    " Global eps mean/med/min/max eps rew:",
                    "{:.3f}/{:.3f}/{:.3f}/{:.3f},".format(
                        np.mean(g_episode_rewards),
                        np.median(g_episode_rewards),
                        np.min(g_episode_rewards),
                        np.max(g_episode_rewards))
                ])

            if args.eval:
                total_success = []
                total_spl = []
                total_dist = []
                for e in range(args.num_processes):

                    for acc in episode_success[e]:
                        total_success.append(acc)
                    for dist in episode_dist[e]:
                        total_dist.append(dist)
                    for spl in episode_spl[e]:
                        total_spl.append(spl)

                if len(total_spl) > 0:
                    log += " ObjectNav succ/spl/dtg:"
                    log += " {:.3f}/{:.3f}/{:.3f}({:.0f}),".format(
                        np.mean(total_success),
                        np.mean(total_spl),
                        np.mean(total_dist),
                        len(total_spl))
            else:
                if len(episode_success) > 100:
                    log += " ObjectNav succ/spl/dtg:"

                    log_wr.writer.add_scalar("train/statistic/succ", np.mean(episode_success) , step)
                    log_wr.writer.add_scalar("train/statistic/spl", np.mean(episode_spl), step)
                    log_wr.writer.add_scalar("train/statistic/dtg", np.mean(episode_dist) , step)
                    log_wr.writer.add_scalar("train/statistic/softspl", np.mean(episode_softspl) , step)
                    log_wr.writer.add_scalar("train/statistic/agent_success", np.mean(episode_agent_success) , step)


                    log += " {:.3f}/{:.3f}/{:.3f}({:.0f}),".format(
                        np.mean(episode_success),
                        np.mean(episode_spl),
                        np.mean(episode_dist),
                        np.mean(episode_softspl),
                        np.mean(episode_agent_success),
                        len(episode_spl))

            log += "\n\tLosses:"
            if len(explore_value_losses) > 0 and not args.eval:
                log_wr.writer.add_scalar("train/loss/explore_value", np.mean(explore_value_losses) , step)
                log_wr.writer.add_scalar("train/loss/explore_action", np.mean(explore_action_losses) , step)
                log_wr.writer.add_scalar("train/loss/exploredist", np.mean(explore_dist_entropies) , step)

                log_wr.writer.add_scalar("train/loss/identify_value", np.mean(identify_value_losses) , step)
                log_wr.writer.add_scalar("train/loss/identify_action", np.mean(identify_action_losses) , step)
                log_wr.writer.add_scalar("train/loss/identify_dist", np.mean(identify_dist_entropies) , step)


                log += " ".join([
                    " Policy explore Loss value/action/dist:",
                    "{:.3f}/{:.3f}/{:.3f},".format(
                        np.mean(explore_value_losses),
                        np.mean(explore_action_losses),
                        np.mean(explore_dist_entropies))
                ])

                log += " ".join([
                    " Policy identify Loss value/action/dist:",
                    "{:.3f}/{:.3f}/{:.3f},".format(
                        np.mean(identify_value_losses),
                        np.mean(identify_action_losses),
                        np.mean(identify_dist_entropies))
                ])


            print(log)
            logging.info(log)

        # Save best models
        if (step * num_scenes) % args.save_interval < \
                num_scenes:
            if len(g_episode_rewards) >= 1000 and \
                    (np.mean(g_episode_rewards) >= best_g_reward) \
                    and not args.eval:
                torch.save(explore_policy.state_dict(),
                           os.path.join(log_dir, "model_best_explore.pth"))

                torch.save(identify_policy.state_dict(),
                           os.path.join(log_dir, "model_best_identify.pth"))
                best_g_reward = np.mean(g_episode_rewards)

        # Save periodic models
        if (step * num_scenes) % args.save_periodic < \
                num_scenes:
            total_steps = step * num_scenes
            if not args.eval:
                torch.save(explore_policy.state_dict(),
                           os.path.join(dump_dir,
                                        "periodic_explore_{}.pth".format(total_steps)))

                torch.save(identify_policy.state_dict(),
                           os.path.join(dump_dir,
                                        "periodic_identify_{}.pth".format(total_steps)))


    # Print and save model performance numbers during evaluation
    if args.eval:
        print("Dumping eval details...")
        
        total_success = []
        total_spl = []
        total_dist = []
        for e in range(args.num_processes):
            for acc in episode_success[e]:
                total_success.append(acc)
            for dist in episode_dist[e]:
                total_dist.append(dist)
            for spl in episode_spl[e]:
                total_spl.append(spl)

        if len(total_spl) > 0:
            log = "Final ObjectNav succ/spl/dtg:"
            log += " {:.3f}/{:.3f}/{:.3f}({:.0f}),".format(
                np.mean(total_success),
                np.mean(total_spl),
                np.mean(total_dist),
                len(total_spl))

        print(log)
        logging.info(log)
            
        # Save the spl per category
        log = "Success | SPL per category\n"
        for key in success_per_category:
            log += "{}: {} | {}\n".format(key,
                                          sum(success_per_category[key]) /
                                          len(success_per_category[key]),
                                          sum(spl_per_category[key]) /
                                          len(spl_per_category[key]))

        print(log)
        logging.info(log)

        with open('{}/{}_spl_per_cat_pred_thr.json'.format(
                dump_dir, args.split), 'w') as f:
            json.dump(spl_per_category, f)

        with open('{}/{}_success_per_cat_pred_thr.json'.format(
                dump_dir, args.split), 'w') as f:
            json.dump(success_per_category, f)


if __name__ == "__main__":
    main()
