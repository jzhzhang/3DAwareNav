import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import cv2
from utils.distributions import Categorical, DiagGaussian
from utils.model import get_grid, ChannelPool, Flatten, NNBase
import envs.utils.depth_utils as du
from utils.pointnet import PointNetEncoder
from utils.ply import write_ply_xyz, write_ply_xyz_rgb
# from pytorch3d.ops import sample_farthest_points





class Goal_Oriented_Semantic_Policy(NNBase):

    def __init__(self, input_map_shape, input_points_shape, recurrent=False, hidden_size=512,
                 num_sem_categories=6):
        super(Goal_Oriented_Semantic_Policy, self).__init__(
            recurrent, hidden_size, hidden_size)

        out_size = int(input_map_shape[1] / 16.) * int(input_map_shape[2] / 16.)


        self.pointEncoder = PointNetEncoder(global_feat=True, feature_transform=True, channel = 7)

        self.main = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(num_sem_categories + 8, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            Flatten()
        )

        self.linear1 = nn.Linear(out_size * 32 + 8 * 2 + 1024, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 256)
        self.critic_linear = nn.Linear(256, 1)
        self.orientation_emb = nn.Embedding(72, 8)
        self.goal_emb = nn.Embedding(num_sem_categories, 8)
        self.train()

    def forward(self, inputs_map, input_points, rnn_hxs, masks, extras):
        x = self.main(inputs_map)
        points_x, _ , _  = self.pointEncoder(input_points)
        orientation_emb = self.orientation_emb(extras[:, 0])
        goal_emb = self.goal_emb(extras[:, 1])

        # print(x.shape)
        # print(orientation_emb.shape)
        # print(goal_emb.shape)
        # print(points_x.shape)


        x = torch.cat((x, orientation_emb, goal_emb, points_x), 1)
        # print("shape", x.shape)
        # print("learning========================================================")
        x = nn.ReLU()(self.linear1(x))
        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        x = nn.ReLU()(self.linear2(x))

        return self.critic_linear(x).squeeze(-1), x, rnn_hxs


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
            return self.network(inputs_map, inputs_points, rnn_hxs, masks)
        else:
            return self.network(inputs_map, inputs_points, rnn_hxs, masks, extras)

    def act(self, inputs_map, inputs_points, rnn_hxs, masks, extras=None, deterministic=False):

        value, actor_features, rnn_hxs = self(inputs_map, inputs_points, rnn_hxs, masks, extras)
        dist = self.dist(actor_features)

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


        self.save_points_count = 0

    def forward(self, obs, pose_obs, maps_last, poses_last, origins, full_map_points, goal_cat_id):
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

        agent_view = torch.zeros(bs, c - 1,
                                 self.map_size_cm // self.resolution,
                                 self.map_size_cm // self.resolution
                                 ).to(self.device)

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

        # points_current_poses = current_poses.clone()
        # points_current_poses[:,2] =points_current_poses[:,2] * np.pi /180 
        # points_current_poses[:,:2] = points_current_poses[:,:2] * 100
        # print("points_current_poses: ", points_current_poses)

        # points_pose = current_poses.clone()+torch.from_numpy(origins).to(self.device)
        # print("orgins: ", origins)
        # print("current_poses: ", current_poses)

        points_pose = current_poses.clone()

        points_pose[:,2] =points_pose[:,2] * np.pi /180 
        points_pose[:,:2] = points_pose[:,:2] * 100
        robot_location = points_pose.clone()
        robot_location[:, 2] = self.agent_height
        # print(robot_location)
        # print("points_pose: ", points_pose)

        # print("agent_view_t_3d",agent_view_t_3d.shape)
        # print("points_pose",points_pose.shape)

        for e in range(bs):
            # world_view_t = du.transform_pose_t2(
            #     agent_view_t_3d, points_current_poses[e,...].cpu().numpy(), self.device).reshape(-1,3)

            world_view_t = du.transform_pose_t2(
                agent_view_t_3d[e,...], points_pose[e,...].cpu().numpy(), self.device).reshape(-1,3)


            # print("world_view shape",world_view_t.shape)
            non_zero_row = torch.abs(point_cloud_t_3d[e,...].reshape(-1,3)).sum(dim=1) > 0

            # print("non_zero_row", non_zero_row)
            # print("non_zero_row",non_zero_row.shape)


            world_view_rgb = obs[e, :3, :, :].permute(1,2,0).reshape(-1,3)
            world_view_rgb = world_view_rgb[non_zero_row].transpose(1,0)

            # print("rgb", world_view_rgb.shape)

            world_view_entropy = obs[e, -1, :, :].reshape(-1)
            world_view_entropy = world_view_entropy[non_zero_row]
            # print("entropy", world_view_entropy.shape)
            # print("max entropy", torch.max(world_view_entropy))

            # world_view_t = world_view_t[e, non_zero_row].transpose(1,0) / self.resolution
            # world_view_t = (world_view_t - robot_location).transpose(1,0)


            # new obs view feature
            wolrd_points = torch.cat((world_view_t[non_zero_row].transpose(1,0) / self.resolution, world_view_rgb, world_view_entropy[None,:]), 0)
            # print("wolrd_points shape", wolrd_points.shape)

            # print("full_map_points_1",full_map_points[:,-1,:])


            if wolrd_points.shape[1]<4096:
                continue
            
                # write_ply_xyz_rgb(world_view_t.transpose(1,0).cpu().numpy(), world_view_rgb.transpose(1,0).cpu().numpy(), "/home/jiazhao/code/navigation/results/world_view_rgb_"+str(self.save_points_count)+".ply")
            # elif torch.max(full_map_points[e, ...]) == 0:
                # print("init frame")
                # world_view_entropy 
            current_view_idx = world_view_entropy.multinomial(num_samples=4096)
            # indices = torch.randperm(wolrd_points.shape[1])[:4096]
            # full_map_points[e, ...] = wolrd_points[:,idx]

            if torch.max(full_map_points[e, ...]) != 0:

                tmp_full_map_points = torch.cat( (full_map_points[e,:,:], wolrd_points[:, current_view_idx]), 1)
                idx = tmp_full_map_points[-1, :].multinomial(num_samples=4096)
                full_map_points[e, ...] = tmp_full_map_points[:, idx]
                # indices = torch.randperm(wolrd_points.shape[1])[:2048]
                # indices_full_map = torch.randperm(full_map_points.shape[2])[:2048]
                # full_map_points[e, ...] = torch.cat( (full_map_points[e,:,indices_full_map] ,wolrd_points[:,indices]),1)
            else:
                full_map_points[e, ...] = wolrd_points[:, current_view_idx]

            # print("full_map_points_2",full_map_points[:,-1,:])


            # cv2.imwrite("/home/jiazhao/code/navigation/results/rgb/world_view_rgb_"+str(self.save_points_count)+".png",  cv2.cvtColor(obs[e, :3, :, :].permute(1,2,0).cpu().numpy(),cv2.COLOR_RGB2BGR))
            # cv2.imwrite("/home/jiazhao/code/navigation/results/depth/world_view_depth_"+str(self.save_points_count)+".png", depth.permute(1,2,0).cpu().numpy())
            
            # if self.save_points_count%2 == 0:
            #     write_ply_xyz_rgb(full_map_points[0,:3,:].transpose(1,0).cpu().numpy(), full_map_points[0,3:6,:].transpose(1,0).cpu().numpy(), "/home/jiazhaozhang/project/navigation/Object-Goal-Navigation_3D_points/tmp/points/world_view_rgb_"+str(self.save_points_count)+".ply")
            # if self.save_points_count%2 == 0:
            #     write_ply_xyz_rgb(full_map_points[e,:3,:].transpose(1,0).cpu().numpy(), full_map_points[e,3:,:].transpose(1,0).cpu().numpy(), "/home/jiazhao/code/navigation/results/points/world_view_rgb_"+str(self.save_points_count)+".ply")


        obs_map_points = full_map_points.clone() 
        obs_map_points[:,:3,:] = obs_map_points[:,:3,:] - robot_location[:,:,None] / self.resolution

        # if self.save_points_count%2 == 0:
        #     write_ply_xyz_rgb(obs_map_points[e,:3,:].transpose(1,0).cpu().numpy(), obs_map_points[e,3:,:].transpose(1,0).cpu().numpy(), "/home/jiazhao/code/navigation/results/points/world_view_rgb_"+str(self.save_points_count)+".ply")


        # print( "rotated shape", rotated.shape)

        # print( "translated shape", translated.shape)
        # write_ply_xyz(translated.cpu().numpy()[0,:3,...].transpose(1, 2, 0).reshape(-1,3),"/home/jiazhao/code/navigation/results/translated"+str(self.save_points_count)+".ply")
        self.save_points_count+=1

        maps2 = torch.cat((maps_last.unsqueeze(1), translated.unsqueeze(1)), 1)

        map_pred, _ = torch.max(maps2, 1)

        return fp_map_pred, map_pred, pose_pred, current_poses, full_map_points , obs_map_points
