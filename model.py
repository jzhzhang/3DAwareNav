import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_scatter import scatter
import numpy as np
import cv2
from utils.distributions import Categorical, DiagGaussian
from utils.model import get_grid, ChannelPool, Flatten, NNBase
import envs.utils.depth_utils as du
from utils.pointnet import PointNetEncoder
from utils.ply import write_ply_xyz, write_ply_xyz_rgb
from utils.img_save import save_semantic, save_KLdiv
from arguments import get_args

import os
import time
# from pytorch3d.ops import sample_farthest_points

# 3DV

'''
    # another method
    for i in range(input_points.shape[0]) : # number of agent 
        for p in range(input_points.shape[2]) : # number of point (4096)
            #print(type(input_points[i,10,p].item()))
            if int(input_points[i,0,p]) == -360 and int(input_points[i,1,p]) == -360 :
                break
            if int(input_points[i,0,p]) < 0 or int(input_points[i,0,p]) >= 240 or \
                int(input_points[i,1,p]) < 0 or int(input_points[i,1,p]) >= 240 :
                continue
            points_map[i, 0, int(input_points[i,0,p]), int(input_points[i,1,p])] += input_points[i,10,p].item()
            points_map_cnt[i, 0, int(input_points[i,0,p]), int(input_points[i,1,p])] += 1.
    save_KLdiv("./tmp/points/map1/time_{0}.png".format(str(time.time()).replace('.','')), points_map)
    points_map = torch.div(points_map, points_map_cnt)
    points_map = torch.where(torch.isnan(points_map), torch.full_like(points_map, 0), points_map)
    torch.set_printoptions(profile="full")
    with open("./tmp/points/map1/time_{0}.txt".format(str(time.time()).replace('.','')), "w") as external_file:
        print(points_map, file=external_file)
        external_file.close()
    torch.set_printoptions(profile="default")
'''

class Goal_Oriented_Semantic_Policy(NNBase):

    def __init__(self, input_map_shape, input_points_shape, recurrent = False, hidden_size = 512,
                 num_sem_categories = 6):
        super(Goal_Oriented_Semantic_Policy, self).__init__(
            recurrent, hidden_size, hidden_size)

        self.in_size_x = input_map_shape[1]
        self.in_size_y = input_map_shape[2]
        self.out_size_x = int(self.in_size_x / 16.) 
        self.out_size_y = int(self.in_size_y / 16.)

        self.layer_attached = 0
        args = get_args()
        if args.deactivate_klmap == False :
            self.layer_attached += 1
        if args.deactivate_entropymap == False :
            self.layer_attached += 1
        
        self.policy_net = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(num_sem_categories + 8 + self.layer_attached, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            Flatten()
        )

        self.rnn_mlp1 = nn.Sequential(
            nn.Linear(self.out_size_x * self.out_size_y * 32, hidden_size),
            nn.ReLU()
        )
        self.rnn_mlp2 = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU()
        )

        self.emb_mlp1 = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU()
        )
        self.emb_mlp2 = nn.Sequential(
            nn.Linear(64 + 24 , 256),
            nn.ReLU()
        )
        self.orientation_emb = nn.Embedding(72, 8)
        self.goal_emb = nn.Embedding(num_sem_categories, 8)
        self.time_emb = nn.Embedding(500, 8)

        self.critic_mlp = nn.Linear(256, 1)

        self.train()

    def forward(self, inputs_map, input_points, rnn_hxs, masks, extras):
        # T1 = time.time()
        # batch size
        bs = inputs_map.shape[0]

        x = inputs_map

        # 3D points information
        args = get_args()

        # KL_Divergency Map
        if args.deactivate_klmap == False :
            points_map = torch.zeros([inputs_map.shape[0], 1, self.in_size_x, self.in_size_y], dtype=torch.float).to(x.device)
            for p in range(bs) :
                # filter the point
                input_points_ful = input_points[p].transpose(1, 0)
                input_points_ful = input_points_ful[ torch.where( (input_points_ful[:, 0] >= 0) & \
                    (input_points_ful[:, 0] < self.in_size_x) & (input_points_ful[:, 1] >= 0) & \
                    (input_points_ful[:, 1] < self.in_size_y) )]
                
                # get the index
                input_points_pos = input_points_ful[:, :2].long()
                points_map_index = input_points_pos[:, 1] * int(self.in_size_y) + input_points_pos[:, 0]        
                point_cnt = torch.count_nonzero(points_map_index).item()
                if point_cnt == 0 :
                    continue
                
                # get the value
                points_map_value = input_points_ful[:, -2].clamp(max=1.0)#.reshape(input_points_ful.shape[0])

                # scatter the value and normalization
                points_map_tmp = scatter(points_map_value, points_map_index, dim=0, reduce='mean')
                points_map_tmp_extend = torch.zeros([self.in_size_x * self.in_size_y - points_map_tmp.shape[0]], \
                    dtype=torch.float).to(points_map_tmp.device)
                points_map_tmp = torch.cat((points_map_tmp, points_map_tmp_extend), 0).reshape(1, self.in_size_x, self.in_size_y)

                points_map[p, 0] = points_map_tmp
            
            x = torch.cat((x, points_map), 1)
        
        # Entropy Map
        if args.deactivate_entropymap == False :
            points_map = torch.zeros([inputs_map.shape[0], 1, self.in_size_x, self.in_size_y], dtype=torch.float).to(x.device)
            for p in range(bs) :
                # filter the point
                input_points_ful = input_points[p].transpose(1, 0)
                input_points_ful = input_points_ful[ torch.where( (input_points_ful[:, 0] >= 0) & \
                    (input_points_ful[:, 0] < self.in_size_x) & (input_points_ful[:, 1] >= 0) & \
                    (input_points_ful[:, 1] < self.in_size_y) )]
                
                # get the index
                input_points_pos = input_points_ful[:, :2].long()
                points_map_index = input_points_pos[:, 1] * int(self.in_size_y) + input_points_pos[:, 0]        
                point_cnt = torch.count_nonzero(points_map_index).item()
                if point_cnt == 0 :
                    continue
                
                # get the value
                points_map_value = input_points_ful[:, -1]#.reshape(input_points_ful.shape[0])

                # scatter the value and normalization
                points_map_tmp = scatter(points_map_value, points_map_index, dim=0, reduce='mean')
                points_map_tmp_extend = torch.zeros([self.in_size_x * self.in_size_y - points_map_tmp.shape[0]], \
                    dtype=torch.float).to(points_map_tmp.device)
                points_map_tmp = torch.cat((points_map_tmp, points_map_tmp_extend), 0).reshape(1, self.in_size_x, self.in_size_y)

                points_map[p, 0] = points_map_tmp
            
            x = torch.cat((x, points_map), 1)
        
        # T2 = time.time()
        # time1 = (T2-T1)*1000
        # print("run time1: "+str(time1)+" ms")

        # policy net
        x = self.policy_net(x)

        # RNN module (deactive)
        x = self.rnn_mlp1(x)
        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)
        x = self.rnn_mlp2(x)
        
        # extra information embedding
        #   output: bs * (4 * 3 = 12)
        x = self.emb_mlp1(x)

        orientation_emb = self.orientation_emb(extras[:, 0])
        goal_emb = self.goal_emb(extras[:, 1])
        time_effe_emb = self.time_emb(extras[:, 2])
        extra_tot = torch.cat((orientation_emb, goal_emb, time_effe_emb), 1)
        x = torch.cat((x, extra_tot), 1)

        x = self.emb_mlp2(x)
        
        return self.critic_mlp(x).squeeze(-1), x, rnn_hxs


# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/model.py#L15
class RL_Policy(nn.Module):

    def __init__(self, obs_map_shape, obs_points_shape, action_space, model_type=0,
                 base_kwargs=None):

        super(RL_Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}

        if model_type == 1:
            self.network = Goal_Oriented_Semantic_Policy(
                obs_map_shape, obs_points_shape, **base_kwargs)
        else:
            raise NotImplementedError

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.network.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.network.output_size, num_outputs)
        else:
            raise NotImplementedError

        self.model_type = model_type

    @property
    def is_recurrent(self):
        return self.network.is_recurrent

    @property
    def rec_state_size(self):
        """Size of rnn_hx."""
        return self.network.rec_state_size

    def forward(self, inputs_map, inputs_points, rnn_hxs, masks, extras):
        if extras is None:
            return self.network(inputs_map, inputs_points , rnn_hxs, masks)
        else:
            return self.network(inputs_map, inputs_points, rnn_hxs, masks, extras)

    def act(self, inputs_map, inputs_points, rnn_hxs, masks, extras=None, deterministic=False):

        value, actor_features, rnn_hxs = self(inputs_map, inputs_points, rnn_hxs, masks, extras)
        #torch.set_printoptions(profile='full')
        #print("actor:", actor_features)
        #print(actor_features.shape)
        #torch.set_printoptions(profile='default')
        dist = self.dist(actor_features)
        #print(type(dist))

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs_map, inputs_points, rnn_hxs, masks, extras=None):
        value, _, _ = self(inputs_map, inputs_points, rnn_hxs, masks, extras)
        return value

    def evaluate_actions(self, inputs_map, inputs_points, rnn_hxs, masks, action, extras=None):

        value, actor_features, rnn_hxs = self(inputs_map, inputs_points, rnn_hxs, masks, extras)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


class Semantic_Mapping(nn.Module):

    """
    Semantic_Mapping
    """

    def __init__(self, args):
        super(Semantic_Mapping, self).__init__()
        # print(args.device)
        # exit(0)
        self.device = args.device
        self.screen_h = args.frame_height
        self.screen_w = args.frame_width
        self.resolution = args.map_resolution
        self.z_resolution = args.map_resolution
        self.map_size_cm = args.map_size_cm // args.global_downscaling
        self.n_channels = 3
        self.vision_range = args.vision_range
        self.dropout = 0.5
        self.fov = args.hfov
        self.du_scale = args.du_scale
        self.cat_pred_threshold = args.cat_pred_threshold
        self.exp_pred_threshold = args.exp_pred_threshold
        self.map_pred_threshold = args.map_pred_threshold
        self.num_sem_categories = args.num_sem_categories

        self.max_height = int(360 / self.z_resolution)
        self.min_height = int(-40 / self.z_resolution)
        self.agent_height = args.camera_height * 100.
        self.shift_loc = [self.vision_range *
                          self.resolution // 2, 0, np.pi / 2.0]
        self.camera_matrix = du.get_camera_matrix(
            self.screen_w, self.screen_h, self.fov)

        self.pool = ChannelPool(1)

        vr = self.vision_range

        self.init_grid = torch.zeros(
            args.num_processes, 1 + self.num_sem_categories, vr, vr,
            self.max_height - self.min_height
        ).float().to(self.device)
        self.feat = torch.ones(
            args.num_processes, 1 + self.num_sem_categories,
            self.screen_h // self.du_scale * self.screen_w // self.du_scale
        ).float().to(self.device)

        # logging information ##############################
        # self.mapping_infos=[]
        # for _ in range(args.num_processes):
        #     m_info = dict()
        #     m_info["timestep"] = 0
        #     m_info["seq_num"] = 0
        #     self.mapping_infos.append(m_info)




    def forward(self, obs, pose_obs, maps_last, poses_last, origins, observation_points, goal_cat_id, gl_tree_list, infos, wait_env, args):

        # print(wait_env)

        bs, c, h, w = obs.size()
        depth = obs[:, 3, :, :]


        # depth[depth>500] =0

        point_cloud_t = du.get_point_cloud_from_z_t(
            depth, self.camera_matrix, self.device, scale=self.du_scale)

        point_cloud_t_3d = point_cloud_t.clone()


        agent_view_t = du.transform_camera_view_t(
            point_cloud_t, self.agent_height, 0, self.device)
        
        agent_view_t_3d = point_cloud_t.clone()


        agent_view_centered_t = du.transform_pose_t(
            agent_view_t, self.shift_loc, self.device)


        max_h = self.max_height
        min_h = self.min_height
        xy_resolution = self.resolution
        z_resolution = self.z_resolution
        vision_range = self.vision_range
        XYZ_cm_std = agent_view_centered_t.float()
        XYZ_cm_std[..., :2] = (XYZ_cm_std[..., :2] / xy_resolution)
        XYZ_cm_std[..., :2] = (XYZ_cm_std[..., :2] -
                               vision_range // 2.) / vision_range * 2.
        XYZ_cm_std[..., 2] = XYZ_cm_std[..., 2] / z_resolution
        XYZ_cm_std[..., 2] = (XYZ_cm_std[..., 2] -
                              (max_h + min_h) // 2.) / (max_h - min_h) * 2.

        # print("sem", obs[:, 4:4+(self.num_sem_categories), :, :])

        self.feat[:, 1:, :] = nn.AvgPool2d(self.du_scale)(
            obs[:, 4:4+(self.num_sem_categories), :, :]
        ).view(bs, self.num_sem_categories, h // self.du_scale * w // self.du_scale)

        XYZ_cm_std = XYZ_cm_std.permute(0, 3, 1, 2)
        XYZ_cm_std = XYZ_cm_std.view(XYZ_cm_std.shape[0],
                                     XYZ_cm_std.shape[1],
                                     XYZ_cm_std.shape[2] * XYZ_cm_std.shape[3])

        voxels = du.splat_feat_nd(
            self.init_grid * 0., self.feat, XYZ_cm_std).transpose(2, 3)

        # print("voxels", voxels.shape)


        min_z = int(25 / z_resolution - min_h)
        max_z = int((self.agent_height + 1) / z_resolution - min_h)

        agent_height_proj = voxels[..., min_z:max_z].sum(4)
        all_height_proj = voxels.sum(4)

        fp_map_pred = agent_height_proj[:, 0:1, :, :]
        fp_exp_pred = all_height_proj[:, 0:1, :, :]
        fp_map_pred = fp_map_pred / self.map_pred_threshold
        fp_exp_pred = fp_exp_pred / self.exp_pred_threshold
        fp_map_pred = torch.clamp(fp_map_pred, min=0.0, max=1.0)
        fp_exp_pred = torch.clamp(fp_exp_pred, min=0.0, max=1.0)

        pose_pred = poses_last

        agent_view = torch.zeros(bs, c - 2,     
                                 self.map_size_cm // self.resolution,
                                 self.map_size_cm // self.resolution
                                 ).to(self.device)  # -2 including, entropy, goal

        x1 = self.map_size_cm // (self.resolution * 2) - self.vision_range // 2
        x2 = x1 + self.vision_range
        y1 = self.map_size_cm // (self.resolution * 2)
        y2 = y1 + self.vision_range
        agent_view[:, 0:1, y1:y2, x1:x2] = fp_map_pred
        agent_view[:, 1:2, y1:y2, x1:x2] = fp_exp_pred


        
        agent_view[:, 4:, y1:y2, x1:x2] = torch.clamp(
            agent_height_proj[:, 1:, :, :] / self.cat_pred_threshold,
            min=0.0, max=1.0)

        corrected_pose = pose_obs

        def get_new_pose_batch(pose, rel_pose_change):

            pose[:, 1] += rel_pose_change[:, 0] * \
                torch.sin(pose[:, 2] / 57.29577951308232) \
                + rel_pose_change[:, 1] * \
                torch.cos(pose[:, 2] / 57.29577951308232)
            pose[:, 0] += rel_pose_change[:, 0] * \
                torch.cos(pose[:, 2] / 57.29577951308232) \
                - rel_pose_change[:, 1] * \
                torch.sin(pose[:, 2] / 57.29577951308232)
            pose[:, 2] += rel_pose_change[:, 2] * 57.29577951308232

            pose[:, 2] = torch.fmod(pose[:, 2] - 180.0, 360.0) + 180.0
            pose[:, 2] = torch.fmod(pose[:, 2] + 180.0, 360.0) - 180.0

            return pose

        current_poses = get_new_pose_batch(poses_last, corrected_pose)
        st_pose = current_poses.clone().detach()
        # print("st_pose0: ", current_poses)

        st_pose[:, :2] = - (st_pose[:, :2]
                            * 100.0 / self.resolution
                            - self.map_size_cm // (self.resolution * 2)) /\
            (self.map_size_cm // (self.resolution * 2))
        st_pose[:, 2] = 90. - (st_pose[:, 2])

        # print("st_pose1: ", st_pose)

        rot_mat, trans_mat = get_grid(st_pose, agent_view.size(),
                                      self.device)

        rotated = F.grid_sample(agent_view, rot_mat, align_corners=True)
        translated = F.grid_sample(rotated, trans_mat, align_corners=True)

        points_pose = current_poses.clone()
        points_pose[:, :2] =  points_pose[:, :2] + torch.from_numpy(origins[:, :2]).to(self.device).float()

        points_pose[:,2] =points_pose[:,2] * np.pi /180 
        points_pose[:,:2] = points_pose[:,:2] * 100

        goal_maps = torch.zeros([bs, 1, 240, 240],dtype=float)

        import time
        for e in range(bs):
            # if str(infos[e]["episode_no"]) not in ['7']:
            #     continue
            #if str(infos[e]["episode_no"]) == '14':
            #     print(cut)
            #if str(infos[e]["episode_no"]) == '10':
            #     exit(0)
            # if wait_env[e]:
            #     continue

            time_s = time.time()

            world_view_t = du.transform_pose_t2(
                agent_view_t_3d[e,...], points_pose[e,...].cpu().numpy(), self.device).reshape(-1,3)

            world_view_sem_t = obs[e, 4:4+(self.num_sem_categories), :, :].reshape((self.num_sem_categories), -1).transpose(0, 1)

            non_zero_row_1 = torch.abs(point_cloud_t_3d[e,...].reshape(-1,3)).sum(dim=1) > 0
            non_zero_row_2 = torch.abs(world_view_sem_t).sum(dim=1) > 0
            non_zero_row_3 = torch.argmax(world_view_sem_t, dim=1) != 6

            non_zero_row = non_zero_row_1 & non_zero_row_2 & non_zero_row_3
            world_view_sem = world_view_sem_t[non_zero_row].cpu().numpy()

            # print("world_view_sem", world_view_sem.shape)
            if world_view_sem.shape[0] <50:
                continue

            world_view_label = np.argmax(world_view_sem, axis=1) # 


            world_view_rgb = obs[e, :3, :, :].permute(1,2,0).reshape(-1,3)[non_zero_row].cpu().numpy()
            world_view_t = world_view_t[non_zero_row].cpu().numpy()


            if world_view_t.shape[0] >= 512:
                indx = np.random.choice(world_view_t.shape[0], 512, replace = False)
            else:
                indx = np.linspace(0, world_view_t.shape[0]-1, world_view_t.shape[0]).astype(np.int32)

            gl_tree = gl_tree_list[e]
            gl_tree.init_points_node(world_view_t[indx])
            per_frame_nodes = gl_tree.add_points(world_view_t[indx], world_view_sem[indx], world_view_rgb[indx], world_view_label[indx], infos[e]['timestep'])
            scene_nodes = gl_tree.all_points()
            gl_tree.update_neighbor_points(per_frame_nodes)


            sample_points_tensor = torch.tensor(gl_tree.sample_points())   # local map


            sample_points_tensor[:,:2] = sample_points_tensor[:,:2] - origins[e, :2] * 100
            sample_points_tensor[:, 2]  = sample_points_tensor[:, 2] - 0.88 * 100
            sample_points_tensor[:,:3] = sample_points_tensor[:,:3] / args.map_resolution

            observation_points[e] = sample_points_tensor.transpose(1, 0)



            #======================= visualize =====================
            # points_dir = 'tmp/points/{}/episodes/thread_{}/eps_{}/'.format(
            #     args.exp_name, infos[e]['rank'], infos[e]["episode_no"])

            # os.makedirs(points_dir,exist_ok=True)

            # cv2.imwrite(points_dir+"rank_{0}_eps_{1}_step_{2}_mask.png".format(infos[e]['rank'], infos[e]['episode_no'], infos[e]["timestep"]), mask_map)


            # write_ply_xyz(sample_points_tensor.cpu().numpy(), points_dir+"rank_{0}_eps_{1}_step_{2}_xyz.ply".format(infos[e]['rank'], infos[e]["episode_no"], infos[e]["timestep"]) )
            # gl_tree.node_to_points_label_ply(points_dir+"rank_{0}_eps_{1}_step_{2}_label.ply".format(infos[e]['rank'], infos[e]["episode_no"], infos[e]["timestep"]), scene_nodes)

            # gl_tree.node_to_points_prob_ply(points_dir+"rank_{0}_eps_{1}_step_{2}_prob.ply".format(infos[e]['rank'], infos[e]["episode_no"], infos[e]["timestep"]), scene_nodes)

            #gl_tree.node_to_points_kl_ply(points_dir+"rank_{0}_eps_{1}_step_{2}_kldiv.ply".format(infos[e]['rank'], infos[e]["episode_no"], infos[e]["timestep"]), scene_nodes)

            #sem_obs = obs[e, 4:4+(self.num_sem_categories), :, :].permute(1, 2, 0).cpu().numpy()
            #save_KLdiv(points_dir+"rank_{0}_eps_{1}_step_{2}.png".format(infos[e]['rank'], infos[e]["episode_no"], infos[e]["timestep"]), sem_obs)


        maps2 = torch.cat((maps_last.unsqueeze(1), translated.unsqueeze(1)), 1)

        map_pred, _ = torch.max(maps2, 1)

        return fp_map_pred, map_pred, pose_pred, current_poses, observation_points
