import  numpy as np
from queue import Queue
import time
class point3D:
    def __init__(self,point_coor, point_color):
        self.point_coor=point_coor
        self.point_color=point_color
        self.point_seg_list=[]

        self.branch_array=[None, None, None, None, None, None, None, None]
        self.branch_distance=np.full((8),0.15)
        self.frame_id=0
        

    def add_point_seg(self, point_seg):
        self.point_seg_list.append(point_seg)