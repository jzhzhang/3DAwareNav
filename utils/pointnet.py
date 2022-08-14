# https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet_utils.py

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F




class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True,  channel=3):
        super(PointNetEncoder, self).__init__()
        # self.stn = STN3d(channel)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        # self.bn1 = nn.BatchNorm1d(64)
        # self.bn2 = nn.BatchNorm1d(128)
        # self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        # self.feature_transform = feature_transform
        # if self.feature_transform:
        #     self.fstn = STNkd(k=64)

    def forward(self, x):
        B, D, N = x.size()
        # trans = self.stn(x)
        # x = x.transpose(2, 1)
        # if D > 3:
        #     feature = x[:, :, 3:]
        #     x = x[:, :, :3]
        # # x = torch.bmm(x, trans)
        # if D > 3:
        #     x = torch.cat([x, feature], dim=2)
        # x = x.transpose(2, 1)
        x = F.relu(self.conv1(x))

        # if self.feature_transform:
        #     trans_feat = self.fstn(x)
        #     x = x.transpose(2, 1)
        #     x = torch.bmm(x, trans_feat)
        #     x = x.transpose(2, 1)
        # else:
        #     trans_feat = None

        pointfeat = x
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1)




# class PointNetEncoder_STN(nn.Module):
#     def __init__(self, global_feat=True, channel=3):
#         super(PointNetEncoder_STN, self).__init__()
#         self.stn = STN3d(channel)
#         self.conv1 = torch.nn.Conv1d(channel, 64, 1)
#         self.conv2 = torch.nn.Conv1d(64, 128, 1)
#         self.conv3 = torch.nn.Conv1d(128, 1024, 1)
#         # self.bn1 = nn.BatchNorm1d(64)
#         # self.bn2 = nn.BatchNorm1d(128)
#         # self.bn3 = nn.BatchNorm1d(1024)
#         self.global_feat = global_feat
#         # self.feature_transform = feature_transform
#         # if self.feature_transform:
#         self.fstn = STNkd(k=64)

#     def forward(self, x):
#         B, D, N = x.size()
#         trans = self.stn(x)
#         x = x.transpose(2, 1)
#         if D > 3:
#             feature = x[:, :, 3:]
#             x = x[:, :, :3]
#         # x = torch.bmm(x, trans)
#         if D > 3:
#             x = torch.cat([x, feature], dim=2)
#         x = x.transpose(2, 1)
#         x = F.relu(self.conv1(x))

#         # if self.feature_transform:
#         trans_feat = self.fstn(x)
#         x = x.transpose(2, 1)
#         x = torch.bmm(x, trans_feat)
#         x = x.transpose(2, 1)
#         # else:
#         #     trans_feat = None

#         pointfeat = x
#         x = F.relu(self.conv2(x))
#         x = self.conv3(x)
#         x = torch.max(x, 2, keepdim=True)[0]
#         x = x.view(-1, 1024)
#         if self.global_feat:
#             return x
#         else:
#             x = x.view(-1, 1024, 1).repeat(1, 1, N)
#             return torch.cat([x, pointfeat], 1)



# def feature_transform_reguliarzer(trans):
#     d = trans.size()[1]
#     I = torch.eye(d)[None, :, :]
#     if trans.is_cuda:
#         I = I.to("cuda:0")
#     loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
