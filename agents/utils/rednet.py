# From Vince Cartillier's Semantic MapNet

# Original Rednet
# https://github.com/JindongJiang/RedNet
# Vince's Port
# https://github.com/vincentcartillier/Semantic-MapNet

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import matplotlib.pyplot as plt
# The following code is largely borrowed from
# https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py and
# https://github.com/facebookresearch/detectron2/blob/master/demo/predictor.py

import argparse
import time

import torch
import numpy as np

from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.data.catalog import MetadataCatalog
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.visualizer import ColorMode, Visualizer
import detectron2.data.transforms as T

from constants import coco_categories_mapping, fourty221, habitat_labels


# Resnet model urls
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def debug_tensor(label, tensor):
    print(label, tensor.size(), tensor.mean().item(), tensor.std().item())

class RedNet(nn.Module):
    def __init__(self, num_classes=40, pretrained=False):

        super().__init__()

        block = Bottleneck
        transblock = TransBasicBlock
        layers = [3, 4, 6, 3]
        # original resnet
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # resnet for depth channel
        self.inplanes = 64
        self.conv1_d = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                                 bias=False)
        self.bn1_d = nn.BatchNorm2d(64)
        self.layer1_d = self._make_layer(block, 64, layers[0])
        self.layer2_d = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3_d = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4_d = self._make_layer(block, 512, layers[3], stride=2)

        self.inplanes = 512
        self.deconv1 = self._make_transpose(transblock, 256, 6, stride=2)
        self.deconv2 = self._make_transpose(transblock, 128, 4, stride=2)
        self.deconv3 = self._make_transpose(transblock, 64, 3, stride=2)
        self.deconv4 = self._make_transpose(transblock, 64, 3, stride=2)

        self.agant0 = self._make_agant_layer(64, 64)
        self.agant1 = self._make_agant_layer(64 * 4, 64)
        self.agant2 = self._make_agant_layer(128 * 4, 128)
        self.agant3 = self._make_agant_layer(256 * 4, 256)
        self.agant4 = self._make_agant_layer(512 * 4, 512)

        # final block
        self.inplanes = 64
        self.final_conv = self._make_transpose(transblock, 64, 3)

        self.final_deconv_custom = nn.ConvTranspose2d(self.inplanes, num_classes, kernel_size=2,
                                               stride=2, padding=0, bias=True)

        self.out5_conv_custom = nn.Conv2d(256, num_classes, kernel_size=1, stride=1, bias=True)
        self.out4_conv_custom = nn.Conv2d(128, num_classes, kernel_size=1, stride=1, bias=True)
        self.out3_conv_custom = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, bias=True)
        self.out2_conv_custom = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, bias=True)

        if pretrained:
            self._load_resnet_pretrained()

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()



    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []

        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_transpose(self, block, planes, blocks, stride=1):

        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes,
                                   kernel_size=2, stride=stride,
                                   padding=0, bias=False),
                nn.BatchNorm2d(planes),
            )
        elif self.inplanes != planes:
            upsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []

        for i in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes))

        layers.append(block(self.inplanes, planes, stride, upsample))
        self.inplanes = planes

        return nn.Sequential(*layers)

    def _make_agant_layer(self, inplanes, planes):

        layers = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        return layers

    def _load_resnet_pretrained(self):
        pretrain_dict = model_zoo.load_url(model_urls['resnet50'])
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                if k.startswith('conv1'):  # the first conv_op
                    model_dict[k] = v
                    model_dict[k.replace('conv1', 'conv1_d')] = torch.mean(v, 1).data. \
                        view_as(state_dict[k.replace('conv1', 'conv1_d')])

                elif k.startswith('bn1'):
                    model_dict[k] = v
                    model_dict[k.replace('bn1', 'bn1_d')] = v
                elif k.startswith('layer'):
                    model_dict[k] = v
                    model_dict[k[:6] + '_d' + k[6:]] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

    def forward_downsample(self, rgb, depth):
        x = self.conv1(rgb)
        x = self.bn1(x)
        x = self.relu(x)
        depth = self.conv1_d(depth)
        depth = self.bn1_d(depth)
        depth = self.relu(depth)

        fuse0 = x + depth

        x = self.maxpool(fuse0)
        depth = self.maxpool(depth)

        # block 1
        x = self.layer1(x)
        # debug_tensor('post1', depth)
        # for i, mod in enumerate(self.layer1_d):
        #     depth = mod(depth)
        #     debug_tensor(f'post1d-{i}', depth)

        depth = self.layer1_d(depth)
        # debug_tensor('post1d', depth)

        fuse1 = x + depth
        # block 2
        x = self.layer2(fuse1)
        depth = self.layer2_d(depth)
        fuse2 = x + depth
        # block 3
        x = self.layer3(fuse2)
        depth = self.layer3_d(depth)
        fuse3 = x + depth
        # block 4
        x = self.layer4(fuse3)
        depth = self.layer4_d(depth)
        fuse4 = x + depth

        agant4 = self.agant4(fuse4)

        return fuse0, fuse1, fuse2, fuse3, agant4

    def forward_upsample(self, fuse0, fuse1, fuse2, fuse3, agant4):

        # upsample 1
        x = self.deconv1(agant4)
        if self.training:
            out5 = self.out5_conv_custom(x)
        x = x + self.agant3(fuse3)
        # upsample 2
        x = self.deconv2(x)
        if self.training:
            out4 = self.out4_conv_custom(x)
        x = x + self.agant2(fuse2)
        # upsample 3
        x = self.deconv3(x)
        if self.training:
            out3 = self.out3_conv_custom(x)
        x = x + self.agant1(fuse1)
        # upsample 4
        x = self.deconv4(x)
        if self.training:
            out2 = self.out2_conv_custom(x)
        x = x + self.agant0(fuse0)
        # final
        x = self.final_conv(x)

        last_layer = x

        out = self.final_deconv_custom(x)

        if self.training:
            return out, out2, out3, out4, out5

        return out, last_layer

    def forward(self, rgb, depth):

        fuses = self.forward_downsample(rgb, depth)

        # We only need predictions.
        # features_encoder = fuses[-1]
        # scores, features_lastlayer = self.forward_upsample(*fuses)
        scores, *_ = self.forward_upsample(*fuses)
        # debug_tensor('scores', scores)
        # return features_encoder, features_lastlayer, scores
        return scores


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class TransBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None, **kwargs):
        super(TransBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        if upsample is not None and stride != 1:
            self.conv2 = nn.ConvTranspose2d(inplanes, planes,
                                            kernel_size=3, stride=stride, padding=1,
                                            output_padding=1, bias=False)
        else:
            self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        out = self.relu(out)

        return out

class BatchNormalize(nn.Module):
    r"""
        I can't believe this isn't supported
        https://github.com/pytorch/vision/issues/157
    """

    def __init__(self, mean, std, device):
        super().__init__()
        self.mean = torch.tensor(mean, device=device)[None, :, None, None]
        self.std = torch.tensor(std, device=device)[None, :, None, None]

    def forward(self, x):
        # mean = torch.tensor(self.mean, device=x.device)
        # std = torch.tensor(self.std, device=x.device)
        # mean = self.mean[None, :, None, None]
        # std = self.std[None, :, None, None]
        return (x - self.mean) / self.std
        # x.sub_(self.mean).div_(self.std)
        # return x

class RedNetResizeWrapper(nn.Module):
    def __init__(self, device, resize=True, stabilize=False):
        super().__init__()
        self.rednet = RedNet()
        self.semmap_rgb_norm = BatchNormalize(
            mean=[0.493, 0.468, 0.438],
            std=[0.544, 0.521, 0.499],
            device=device
        )
        self.semmap_depth_norm = BatchNormalize(
            mean=[0.213],
            std=[0.285],
            device=device
        )
        self.pretrained_size = (480, 640)
        self.resize = resize
        self.stabilize = stabilize

    def forward(self, rgb, depth):
        r"""
            Args:
                Raw sensor inputs.
                rgb: B x H=256 x W=256 x 3
                depth: B x H x W x 1
            Returns:
                semantic: drop-in replacement for default semantic sensor. B x H x W  (no channel, for some reason)
        """
        # Not quite sure what depth is produced here. Let's just check
        if self.resize:
            _, og_h, og_w, _ = rgb.size() # b h w c
        rgb = rgb.permute(0, 3, 1, 2)
        depth = depth.permute(0, 3, 1, 2)
        rgb = rgb.float() / 255
        if self.resize:
            rgb = F.interpolate(rgb, self.pretrained_size, mode='bilinear')
            depth = F.interpolate(depth, self.pretrained_size, mode='nearest')

        rgb = self.semmap_rgb_norm(rgb)

        depth_clip = (depth < 1.0).squeeze(1)
        # depth_clip = ((depth < 1.0) & (depth > 0.0)).squeeze(1)
        depth = self.semmap_depth_norm(depth)
        with torch.no_grad():
            scores = self.rednet(rgb, depth) # 1 * 40 * 480 * 640
            pred = (torch.max(scores, 1)[1] ) # B x 480 x 640
            #print("pred shape",pred,pred.shape)
            return scores, pred[0]

        # exit if running this line
        exit(0)

        print("[rednet1] weired running points!!!!")
        if self.stabilize: # downsample tiny
            # Mask out out of depth samples
            pred[~depth_clip] = -1 # 41 is UNK in MP3D, but hab-sim renders with -1
            # pred = F.interpolate(pred.unsqueeze(1), (15, 20), mode='nearest').squeeze(1)
        if self.resize:
            pred = F.interpolate(pred.unsqueeze(1), (og_h, og_w), mode='nearest')

        return pred.long().squeeze(1)


def draw_sem_img(sem_img_channels, sem_mask, rgb, sem_entropy, sem_prob, output_path):

    # sem_img_channels_cpu = sem_img_channels.clone().cpu().numpy()
    sem_mask_cpu = sem_mask.clone().cpu().numpy()
    fig = plt.figure(figsize=(24, 24))
    columns = 3
    rows = 5
    for i in range(0, 6):
        # img = np.random.randint(10, size=(h,w))
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(sem_img_channels[:, :, i])

    fig.add_subplot(rows, columns, 7)
    plt.imshow(sem_mask_cpu)

    fig.add_subplot(rows, columns, 8)
    plt.imshow(rgb)

    fig.add_subplot(rows, columns, 9)
    plt.imshow(sem_entropy)

    fig.add_subplot(rows, columns, 10)
    plt.imshow(sem_prob)

    plt.savefig(output_path)
    # plt.show()


def load_rednet(device, ckpt="", resize=True, stabilize=False):
    if not os.path.isfile(ckpt):
        raise Exception(f"invalid path {ckpt} provided for rednet weights")

    model = RedNetResizeWrapper(device, resize=resize, stabilize=stabilize).to(device)

    print("=> loading RedNet checkpoint '{}'".format(ckpt))
    # if device.type == 'cuda':
    #     checkpoint = torch.load(ckpt, map_location='cpu')
    # else:
    checkpoint = torch.load(ckpt, map_location=lambda storage, loc: storage)

    state_dict = checkpoint['model_state']
    prefix = 'module.'
    state_dict = {
        (k[len(prefix):] if k[:len(prefix)] == prefix else k): v for k, v in state_dict.items()
    }
    model.rednet.load_state_dict(state_dict)
    print("=> loaded checkpoint '{}' (epoch {})"
            .format(ckpt, checkpoint['epoch']))

    return model


class SemanticPredRedNet():

    def __init__(self, args):
        self.segmentation_model = load_rednet(args.sem_gpu_id,ckpt = args.checkpt, resize = True)
        self.segmentation_model.eval()
        self.args = args
        self.all_labels = set()
        self.threshold = 0.8
        self.gt_mask = None
        self.goal_cat = None

    def get_prediction(self, img, depth, cat_goal):
        args = self.args
        #image_list = []
        img = img[np.newaxis, :, :, :]
        depth = depth[np.newaxis, :, :, :]
        #print("input shape is ",img.shape)
        #print(self.args.device)
        img = torch.from_numpy(img).float().to(self.args.sem_gpu_id)
        depth = torch.from_numpy(depth).float().to(self.args.sem_gpu_id)
        output, mask = self.segmentation_model(img,depth)
        # print("mask", mask)
        # print("mask.shape", mask.shape)
        
        # print("output", output)
        # print("output.shape", output.shape)


        #print("output shape is",output.shape)
        output = output[0] # 40*480*640
        # output = output *0.1
        # output[output<self.threshold] = 0 #0.9: 30 1.1: 26

        #=================entropy================
        # print("direct output semantic", output.shape)
        semantic_pred = torch.nn.functional.softmax(output, dim=0).permute(1,2,0).cpu().numpy()
        entropy_tmp = -semantic_pred*np.log(semantic_pred)
        sem_entropy = np.sum(entropy_tmp, axis=2)

        output = torch.nn.functional.softmax(output, dim=0).permute(1,2,0).cpu().numpy()
        output[np.where(np.max(output, axis=-1) < self.threshold)] = 0
        
        semantic_input = np.zeros((img.shape[1], img.shape[2], len(habitat_labels) ))
        
        for i in range(0, 40):
            if i in fourty221.keys():

                j = fourty221[i]
                semantic_input[:, :, j] += output[:,:,i]

        semantic_input[:,:,-1] = 1 - np.sum(semantic_input[:,:,:-1], axis=-1)


        #=================prob points================
        goal_cat_output = semantic_input[:, :, cat_goal] + 1e-5
        goal_cat_output[goal_cat_output<0] = 0


        return semantic_input, sem_entropy, goal_cat_output

    def set_gt_mask(self,gt_mask,goal_cat): # goal cat is as it is in fourty221 mapping
        self.gt_mask = gt_mask
        self.goal_cat = goal_cat

# class QuickSemanticPredRedNet():

#     def __init__(self, args):
#         self.segmentation_model = load_rednet(args.device,ckpt = args.checkpt, resize = True)
#         self.segmentation_model.eval()
#         self.args = args
#         self.threshold = 0.7
#         self.all_labels = set()
#         self.gt_mask = None

#     def get_conflict(self,output,goal_cat,ori_goal):
#         output = torch.clone(output)
#         output[goal_cat] *= 0
#         for i in range(40):
#             if i not in fourty221.keys():
#                 output[i] *= 0
#             else:
#                 j = fourty221[i]
#                 if ori_goal in compatible_dict.keys() and j in compatible_dict[ori_goal]:
#                     output[i] *= 0
#         output,_ = torch.max(output,dim = 0)
#         output *= 0.1
#         output[output < 0.9] = 0
#         return output.cpu().numpy()
#     #only care about objects in blacklist
#     def get_black_white_list(self, output,goal_cat,ori_goal,black_list):
#         if ori_goal not in black_list.keys():
#             return np.zeros(output[0].shape)
#         siz = len(black_list[ori_goal])
#         ans = torch.zeros((siz,output.shape[1],output.shape[2])).to(self.args.device)
#         id = 0
#         for i in black_list[ori_goal]:
#             ans[id] = torch.clone( output[twentyone240[i]])
#             id += 1


#         ans, _ = torch.max(ans, dim=0)
#         ans *= 0.1
#         ans[ans < 0.9] = 0
#         return ans.cpu().numpy()

#     def get_prediction(self, img,depth,goal_cat):
#         ori_goal = goal_cat
#         goal_cat = twentyone240[goal_cat]
#         args = self.args
#         #image_list = []
#         img = img[np.newaxis, :, :, :]
#         depth = depth[np.newaxis, :, :, :]
#         #print("input shape is ",img.shape)
#         #print(self.args.device)
#         img = torch.from_numpy(img).float().to(self.args.device)
#         depth = torch.from_numpy(depth).float().to(self.args.device)
#         output, mask = self.segmentation_model(img,depth)
#         #print("output shape is",output.shape)
#         output = output[0]
#         output[goal_cat] *= 0.1
#         max_score = torch.max(output[goal_cat]) - 0.05
#         if self.threshold > max_score:
#             max_score = self.threshold


#         output[goal_cat][output[goal_cat] < max_score] = 0

#         semantic_input = np.zeros((img.shape[1], img.shape[2], 5 + self.args.use_gt_mask ))
#         semantic_input[:,:,0] = output[goal_cat].cpu().numpy()
#         if self.gt_mask is not None:
#             semantic_input[:,:,4] = self.gt_mask
#         if self.args.record_conflict == 1:
#             semantic_input[:,:,1] = self.get_conflict(output,goal_cat,ori_goal)
#         semantic_input[:,:,2] = self.get_black_white_list(output,goal_cat,ori_goal,black_list)
#         semantic_input[:,:,3] = self.get_black_white_list(output,goal_cat,ori_goal,white_list)
#         return semantic_input

#     def set_gt_mask(self,gt_mask):
#         self.gt_mask = gt_mask


# def compress_sem_map(sem_map):
#     c_map = np.zeros((sem_map.shape[1], sem_map.shape[2]))
#     for i in range(sem_map.shape[0]):
    #     c_map[sem_map[i] > 0.] = i + 1
    # return c_map

