import numpy as np
from queue import Queue
import time
from GLtree.interval_tree import RedBlackTree, Node, BLACK, RED, NIL
import random
from constants import color_palette_array, habitat_labels
import pydensecrf.densecrf as dcrf
from matplotlib import cm
import math
# from constants import

# from utils.ply import


# class point3D:
#     def __init__(self, point_coor, feature_2d, max_octree_threshold):
#         self.point_coor=point_coor
#         self.feature_fuse=feature_2d
#         self.branch_array=[None, None, None, None, None, None, None, None]
#         self.branch_distance=np.full((8),max_octree_threshold)
#         self.result_feature=np.zeros((128))
#         self.pred_result=-1
#         self.frame_id=0
#         self.scan_times=0
#         self.uncertainty=1


   
#     def findNearPoint(self, near_node_num, max_node):

#         neighbor_2dfeature=np.zeros((max_node+1,128))
#         neighbor_node=np.zeros((max_node+1,3))
#         count = 0
#         neighbor_node[count, :] = 0
#         neighbor_2dfeature[count] = self.feature_fuse
#         find_queue = Queue()
#         count += 1

#         for i,node in enumerate(self.branch_array):
#             if node is not None:
#                 neighbor_node[count, :] =  node.point_coor - self.point_coor
#                 neighbor_2dfeature[count] = node.feature_fuse
#                 if node.branch_array[i] is not None:
#                     find_queue.put((i,node.branch_array[i])) 
#                 count += 1

#         while not find_queue.empty() and count<max_node+1:
#             index_node=find_queue.get()
#             node=index_node[1]
#             index=index_node[0]
#             neighbor_node[count, :] =  node.point_coor - self.point_coor
#             neighbor_2dfeature[count] = node.feature_fuse
#             if node.branch_array[index] is not None:
#                 find_queue.put((index,node.branch_array[index]))
#             count+=1
        
#         if count>=near_node_num:
#             sample=np.random.choice(count,near_node_num,replace=False)
#         else:
#             sample=np.random.choice(count,near_node_num,replace=True)
#         sample[0] = 0
#         return neighbor_2dfeature[sample,:].T, neighbor_node[sample,:].T, count

'''
Liu Dai & Fanpeng Meng update in 23 July 2022
1. point3D init: num_sem_categories
2. point3D init: self.seg_prob_fused, self_label_thres
3. function add_point_seg: 
    update prob, update label 
    part1: previous scene 
    part2: structure (on going)
4. keyframe for seg?
'''
class point3D:
    def __init__(self, point_coor, point_color, num_sem_categories = len(habitat_labels)):
        self.point_coor = point_coor
        self.point_color = point_color
        self.point_seg_list = []


        self.seg_prob_fused = np.ones(num_sem_categories, dtype=float)
        self.label_thres = 0.8

        self.kl_div = 0.0
        self.entropy = 0.0

        self.max_prob = 0.0

        self.label = -1
        self.branch_array = [None, None, None, None, None, None, None, None]
        self.branch_distance = np.full((8),15)
        self.frame_id = 0



    def add_point_seg(self, point_seg):
        
        '''
        point_seg : num_sem_categories * 1, probs in one frame
        '''
        activate_3d = True

        # record prob
        self.point_seg_list.append(point_seg)

        # None-3d version
        if activate_3d is False :
            return
        
        #--------------- MAX Fusion ---------------#
        # if self.label == -1: #init
        #     self.label = np.argmax(point_seg.reshape(-1))
        #     self.max_prob = np.max(point_seg.reshape(-1))
        # else: #update
        #     new_frame_max_prob = np.max(point_seg.reshape(-1))
        #     if new_frame_max_prob > self.max_prob:
        #         self.label = np.argmax(point_seg.reshape(-1))
        #         self.max_prob = new_frame_max_prob
        #--------------- MAX Fusion ---------------#

        #--------------- Bayesian Fusion ---------------#
        # update prob
        self.seg_prob_fused *= point_seg.reshape(-1)
        self.seg_prob_fused /= np.sum(self.seg_prob_fused) # Normalization
        # update label
        if np.max(self.seg_prob_fused) > self.label_thres or self.label == -1:
            self.label = np.argmax(self.seg_prob_fused)
        #--------------- Bayesian Fusion ---------------#

        #--------------- No Fusion latest frame---------------#
        # self.label = np.argmax(point_seg.reshape(-1))
        #--------------- No Fusion latest frame ---------------#

        #--------------- No Fusion first frame ---------------#
        # if self.label == -1: #init
        #     self.label = np.argmax(point_seg.reshape(-1))
        #     self.max_prob = np.max(point_seg.reshape(-1))
        #--------------- No Fusion first frame ---------------#
        
        #--------------- Update Entropy ---------------#
        tmp_prob_distribution = self.seg_prob_fused
        tmp_prob_distribution[np.where(tmp_prob_distribution == 0)] = 1 # skip log 0
        self.entropy = - np.sum(self.seg_prob_fused * np.log(tmp_prob_distribution))
        #--------------- Update Entropy ---------------#



class GL_tree:

    def __init__(self, opt):
        self.opt = opt
        self.x_rb_tree = RedBlackTree(opt.interval_size)
        self.y_rb_tree = RedBlackTree(opt.interval_size)
        self.z_rb_tree = RedBlackTree(opt.interval_size)

        self.scene_node = set()

        self.observation_window = set()
        self.observation_window_size = opt.observation_window_size

    def reset_gltree(self):
        del self.x_rb_tree
        del self.y_rb_tree
        del self.z_rb_tree
        del self.scene_node

        self.x_rb_tree = RedBlackTree(self.opt.interval_size)
        self.y_rb_tree = RedBlackTree(self.opt.interval_size)
        self.z_rb_tree = RedBlackTree(self.opt.interval_size)
        self.scene_node = set()
        self.observation_window = set()

    def init_points_node(self, points):
        self.x_tree_node_list = []
        self.y_tree_node_list = []
        self.z_tree_node_list = []

        for p in range(points.shape[0]):
            x_temp_node = self.x_rb_tree.add(points[p,0])
            y_temp_node = self.y_rb_tree.add(points[p,1])
            z_temp_node = self.z_rb_tree.add(points[p,2])

            self.x_tree_node_list.append(x_temp_node)
            self.y_tree_node_list.append(y_temp_node)
            self.z_tree_node_list.append(z_temp_node)

    def add_points(self, points, point_seg, points_color, points_label,frame_index):
        


        # add to the global (to do)
        activate_3d = True
        
        per_image_node_set=set()

        for p in range(points.shape[0]):
            
            x_set_union = self.x_tree_node_list[p].set_list
            y_set_union = self.y_tree_node_list[p].set_list
            z_set_union = self.z_tree_node_list[p].set_list
            set_intersection = x_set_union[0] & y_set_union[0] & z_set_union[0]
            temp_branch = [None, None, None, None, None, None, None, None]
            temp_branch_distance = np.full((8), self.opt.max_octree_threshold)
            is_find_nearest = False
            branch_record = set()
            list_intersection=list(set_intersection)
            random.shuffle(list_intersection)






            for point_iter in list_intersection:
                # distance = np.sum(np.absolute(point_iter.point_coor - points[p,:]))
                distance = np.linalg.norm(point_iter.point_coor - points[p,:])

                # print("distance", distance)
                if distance < self.opt.min_octree_threshold:
                    is_find_nearest = True
                    if frame_index!=point_iter.frame_id:
                        #2D-3D fusion
                        point_iter.add_point_seg(point_seg[p, :])
                        # print("add2!")
                        point_iter.frame_id=frame_index
                        if activate_3d is False :
                            point_iter.label = points_label[p]
                    per_image_node_set.add(point_iter)
                    break
                x = int(point_iter.point_coor[0] >= points[p, 0])
                y = int(point_iter.point_coor[1] >= points[p, 1])
                z = int(point_iter.point_coor[2] >= points[p, 2])
                branch_num= x * 4 + y * 2 + z

                if distance < point_iter.branch_distance[7-branch_num]:
                    branch_record.add((point_iter, 7 - branch_num, distance))
                    # point_iter.add_point_seg(point_seg[p, :])
                    # exit(0)
                    if distance < temp_branch_distance[branch_num]:
                        temp_branch[branch_num] = point_iter
                        temp_branch_distance[branch_num] = distance
                        

            if not is_find_nearest:
                new_3dpoint = point3D(points[p, :], points_color[p, :])
                new_3dpoint.add_point_seg(point_seg[p, :])
                new_3dpoint.frame_id = frame_index
                if activate_3d is False :
                    new_3dpoint.label = points_label[p]
                for point_branch in branch_record:
                    point_branch[0].branch_array[point_branch[1]] = new_3dpoint
                    point_branch[0].branch_distance[point_branch[1]] = point_branch[2]

                new_3dpoint.branch_array = temp_branch
                new_3dpoint.branch_distance = temp_branch_distance
                per_image_node_set.add(new_3dpoint)



                for x_set in x_set_union:
                    x_set.add(new_3dpoint)
                for y_set in y_set_union:
                    y_set.add(new_3dpoint)
                for z_set in z_set_union:
                    z_set.add(new_3dpoint)

            # set_intersection = x_set_union[0] & y_set_union[0] & z_set_union[0]
            # print("len(set_intersection)", len(set_intersection))

        # self.scene_node = self.scene_node.union(per_image_node_set)
        # return per_image_node_set

        self.observation_window = self.observation_window.union(per_image_node_set)

        self.scene_node = self.scene_node.union(per_image_node_set)
        return per_image_node_set

    def all_points(self):
        return self.scene_node


    def crf_points(self, per_image_node_set):

        if len(per_image_node_set) > 0 :

            d = dcrf.DenseCRF(len(per_image_node_set), len(habitat_labels))
            
            U = np.zeros((len(habitat_labels),len(per_image_node_set)), dtype=np.float32) # 7*N
            xyz_pc = np.zeros((len(per_image_node_set),3), dtype=np.float32) # N*3
            feat_pc = np.zeros((len(per_image_node_set),3), dtype=np.float32) # N*3

            origin_label = np.ones(len(per_image_node_set), dtype=int)

            # unary potential
            for index, node in enumerate(per_image_node_set):
                U[ : , index] = node.seg_prob_fused
                xyz_pc[index, : ] = node.point_coor # same shape?
                feat_pc[index, : ] = node.point_color # same shape?

                origin_label[index] = node.label
            
            #print(U)
            U = -np.log(U)
            #print(U)
            d.setUnaryEnergy(U)

            # pairwise potensial
            # gaussian
            p_gaussian = np.zeros((xyz_pc.shape[1], xyz_pc.shape[0]))
            xyz_pc_min, xyz_pc_max = np.min(xyz_pc, axis=0), np.max(xyz_pc, axis=0)
            pp = (xyz_pc - xyz_pc_min) / (xyz_pc_max - xyz_pc_min)
            pp = pp.transpose()
            for i in range(p_gaussian.shape[0]):
                p_gaussian[i, :] == pp[i,:]
            d.addPairwiseEnergy(p_gaussian.astype(np.float32),3)

            # bilateral
            xyz_pc_min, xyz_pc_max = np.min(xyz_pc, axis=0), np.max(xyz_pc, axis=0)
            p_bilateral = np.zeros((xyz_pc.shape[1]+feat_pc.shape[1], feat_pc.shape[0]))

            p_xyz = (xyz_pc - xyz_pc_min) / (xyz_pc_max-xyz_pc_min)
            p_xyz = p_xyz.transpose() # (3,N)

            p_feat = feat_pc
            p_feat = p_feat.transpose() #(C,N)

            p_bilateral[:p_xyz.shape[0]] = p_xyz
            p_bilateral[p_xyz.shape[0]:] = p_feat


            # inference
            Q = d.inference(5)

            # update label
            for index, node in enumerate(per_image_node_set):
                node.seg_prob_fused = probs_matrix[:,index]
                if np.max(node.seg_prob_fused) > node.label_thres:
                    node.label = np.argmax(node.seg_prob_fused)
                #node.label = MAP[index]

        return per_image_node_set

    def sample_points(self):
        if len(self.observation_window) > self.observation_window_size:
            remove_node_list = random.sample(self.observation_window, len(self.observation_window) - self.observation_window_size)
            for node in remove_node_list:
                self.observation_window.remove(node)

        observation_points = np.zeros((4096, 12)) #x,y,z, prob(7), kl_divergency, entropy
        for i, node in enumerate(self.observation_window):
            observation_points[i,:3] = node.point_coor
            observation_points[i,3:10] = node.seg_prob_fused
            observation_points[i,10] = node.kl_div
            observation_points[i,11] = node.entropy
            # print(i,node.kl_div)

        return observation_points


    # simple update node
    def update_neighbor_points(self, per_image_node_set):
        for node in per_image_node_set:
            temp_fused = np.copy(node.seg_prob_fused)

            temp_kl_div_max = node.kl_div
            
            temp_center_prob = node.seg_prob_fused
            temp_center_prob[np.where(temp_center_prob == 0)] = 1
            center_point_kl_prob = node.seg_prob_fused * np.log(temp_center_prob)

            for i in range(8):
                if node.branch_array[i] is not None:
                    temp_fused += node.branch_array[i].seg_prob_fused
                    #---- KL Divergence MAX ---#
                    temp_branch_prob = node.branch_array[i].seg_prob_fused
                    temp_branch_prob[np.where(temp_branch_prob == 0)] = 1
                    branch_point_kl_prob = node.seg_prob_fused * np.log(temp_branch_prob)
                    
                    branch_kl_div = np.sum(center_point_kl_prob - branch_point_kl_prob)
                    if branch_kl_div > temp_kl_div_max:
                        temp_kl_div_max = branch_kl_div
                    #---- KL Divergence MAX ---#

            temp_fused /= np.sum(temp_fused)
            node.seg_prob_fused = temp_fused
            node.label = np.argmax(node.seg_prob_fused)

            node.kl_div = temp_kl_div_max
            
            if math.isinf(node.kl_div):
                print("inf occurs !!!!!!!!!!!!")
                print("inf occurs !!!!!!!!!!!!")
                print("inf occurs !!!!!!!!!!!!")

            #print(node.kl_div)


    def find_object_goal_points(self, node_set, goal_obj_id, threshold):
        goal_list = []
        for node in node_set:
            if node.label != goal_obj_id or node.seg_prob_fused[goal_obj_id] < threshold:
                continue
            count = 0
            for i in range(len(habitat_labels)):
                if node.branch_array[i] is not None and node.branch_array[i].label == goal_obj_id:
                    count += 1 
            if count >2:
                goal_list.append(node)

        if len(goal_list) == 0:
            return None 

        observation_points = np.zeros((len(goal_list), 3)) # x,y,z  
        for i, node in enumerate(goal_list):
            observation_points[i] = node.point_coor

        return observation_points


    def node_to_points_label_ply(self, file_name, point_nodes):

        # point_count=point_cloud.shape[0]
        ply_file = open(file_name, 'w')
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write("element vertex " + str(len(point_nodes)) + "\n")
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")

        ply_file.write("property uchar red\n")
        ply_file.write("property uchar green\n")
        ply_file.write("property uchar blue\n")

        ply_file.write("end_header\n")


        points_list = list(point_nodes)

        for i in range(len(point_nodes)):

            points_coor = points_list[i].point_coor

            ply_file.write(str(points_coor[ 0]) + " " +
                        str(points_coor[1]) + " " +
                        str(points_coor[2]))

            label_id = points_list[i].label + 5
            label_id = label_id if label_id<11 else 1

            if label_id<0:
                print("==================== label id < 0 !!!!")


            rgb = color_palette_array[label_id, :]
            #print(rgb)

            # colormap
            # prob = points_list[i].max_prob # 0~1
            # jet_colormap = cm.get_cmap('jet', 100)
            # rgb = jet_colormap(prob)

            ply_file.write(" "+str(int(rgb[0]*255)) + " " +
                            str(int(rgb[1]*255)) + " " +
                            str(int(rgb[2]*255)))


            ply_file.write("\n")

        ply_file.close()
        print("save rgb result to " + file_name)
            
            
    def node_to_points_prob_ply(self, file_name, point_nodes):
    
        # point_count=point_cloud.shape[0]
        ply_file = open(file_name, 'w')
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write("element vertex " + str(len(point_nodes)) + "\n")
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")

        ply_file.write("property uchar red\n")
        ply_file.write("property uchar green\n")
        ply_file.write("property uchar blue\n")

        ply_file.write("end_header\n")



        points_list = list(point_nodes)

        for i in range(len(point_nodes)):

            points_coor = points_list[i].point_coor

            ply_file.write(str(points_coor[ 0]) + " " +
                        str(points_coor[1]) + " " +
                        str(points_coor[2]))

            label_id = points_list[i].label+5
            label_id = label_id if label_id<11 else 1

            if label_id<0:
                print("==================== label id < 0 !!!!")


            # rgb = color_palette_array[label_id, :]
            #print(rgb)

            # colormap
            prob = points_list[i].max_prob # 0~1
            jet_colormap = cm.get_cmap('jet', 100)
            rgb = jet_colormap(prob)


            ply_file.write(" "+str(int(rgb[0]*255)) + " " +
                            str(int(rgb[1]*255)) + " " +
                            str(int(rgb[2]*255)))


            ply_file.write("\n")

        ply_file.close()

        print("save prob result to " + file_name)


    def node_to_points_kl_ply(self, file_name, point_nodes):
    
        # point_count=point_cloud.shape[0]
        ply_file = open(file_name, 'w')
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write("element vertex " + str(len(point_nodes)) + "\n")
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")

        ply_file.write("property uchar red\n")
        ply_file.write("property uchar green\n")
        ply_file.write("property uchar blue\n")

        ply_file.write("end_header\n")



        points_list = list(point_nodes)

        for i in range(len(point_nodes)):

            points_coor = points_list[i].point_coor

            ply_file.write(str(points_coor[ 0]) + " " +
                        str(points_coor[1]) + " " +
                        str(points_coor[2]))

            label_id = points_list[i].label+5
            label_id = label_id if label_id<11 else 1

            if label_id<0:
                print("==================== label id < 0 !!!!")


            # rgb = color_palette_array[label_id, :]
            #print(rgb)

            # colormap
            prob = points_list[i].kl_div # 0~1
            jet_colormap = cm.get_cmap('jet', 100)
            rgb = jet_colormap(prob)


            ply_file.write(" "+str(int(rgb[0]*255)) + " " +
                            str(int(rgb[1]*255)) + " " +
                            str(int(rgb[2]*255)))


            ply_file.write("\n")

        ply_file.close()

        print("save kl result to " + file_name)

        
    def node_to_points_entro_ply(self, file_name, point_nodes):
    
        # point_count=point_cloud.shape[0]
        ply_file = open(file_name, 'w')
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write("element vertex " + str(len(point_nodes)) + "\n")
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")

        ply_file.write("property uchar red\n")
        ply_file.write("property uchar green\n")
        ply_file.write("property uchar blue\n")

        ply_file.write("end_header\n")



        points_list = list(point_nodes)

        for i in range(len(point_nodes)):

            points_coor = points_list[i].point_coor

            ply_file.write(str(points_coor[ 0]) + " " +
                        str(points_coor[1]) + " " +
                        str(points_coor[2]))

            label_id = points_list[i].label+5
            label_id = label_id if label_id<11 else 1

            if label_id<0:
                print("==================== label id < 0 !!!!")


            # rgb = color_palette_array[label_id, :]
            #print(rgb)

            # colormap
            prob = points_list[i].entropy # 0~1
            jet_colormap = cm.get_cmap('jet', 100)
            rgb = jet_colormap(prob)


            ply_file.write(" "+str(int(rgb[0]*255)) + " " +
                            str(int(rgb[1]*255)) + " " +
                            str(int(rgb[2]*255)))


            ply_file.write("\n")

        ply_file.close()

        print("save entropy result to " + file_name)
            