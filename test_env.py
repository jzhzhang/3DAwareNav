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

from model import RL_Policy, Semantic_Mapping, RL_Policy_3D
from utils.storage import GlobalRolloutStorage, GlobalRolloutStorage_3d
from utils.log_writter import log_writter
from envs import make_vec_envs
from arguments import get_args
import algo

import cv2

import matplotlib.pyplot as plt

from GLtree.interval_tree import RedBlackTree, Node, BLACK, RED, NIL
# from GLtree.octree_point import point3D
from GLtree.octree import GL_tree

os.environ["OMP_NUM_THREADS"] = "1"


def main():

    args = get_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device = torch.device(args.policy_gpu_id if args.cuda else "cpu")


    torch.set_num_threads(1)

    envs = make_vec_envs(args)
    obs, infos = envs.reset()

    count = 0
    print("xxxxx")
    while True:
        envs.verify_action()
        print("STR", str(count))
        count += 1


if __name__ == "__main__":
    main()