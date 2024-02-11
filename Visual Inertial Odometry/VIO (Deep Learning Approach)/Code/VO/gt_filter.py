import numpy as np
import cv2
import os
import time
import csv
from collections import defaultdict, namedtuple

from threading import Thread
# from scipy.spatial.transform import Quaternion

import torch

import csv
import os
import numpy as np
from PIL import Image
import csv
import os
from PIL import Image
import math 

import argparse

def relative_position(pos1, pos2):
    """
    Computes the relative position between two points in 3D space.
    """
    x1, y1, z1 = pos1
    x2, y2, z2 = pos2
    
    # Compute the displacement vector from pos1 to pos2
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    dz = abs(z2 - z1)
    
    return [dx, dy, dz]
    
def quaternion_difference(q1, q2):
    """
    Computes the difference between two quaternions q1 and q2.
    """
    qw1, qx1, qy1, qz1 = q1
    qw2, qx2, qy2, qz2 = q2
    
    # Compute the squared magnitude of q2
    mag2 = qw2**2 + qx2**2 + qy2**2 + qz2**2
    
    # Compute the inverse of q2
    q2_inv = [qw2/mag2, -qx2/mag2, -qy2/mag2, -qz2/mag2]
    
    # Compute the product of q1 and q2_inv
    qw_diff = qw1*q2_inv[0] - qx1*q2_inv[1] - qy1*q2_inv[2] - qz1*q2_inv[3]
    qx_diff = qw1*q2_inv[1] + qx1*q2_inv[0] + qy1*q2_inv[3] - qz1*q2_inv[2]
    qy_diff = qw1*q2_inv[2] - qx1*q2_inv[3] + qy1*q2_inv[0] + qz1*q2_inv[1]
    qz_diff = qw1*q2_inv[3] + qx1*q2_inv[2] - qy1*q2_inv[1] + qz1*q2_inv[0]
    
    return [qw_diff, qx_diff, qy_diff, qz_diff]


# read the data in filtered csv
data =[]
gt_file = os.path.join('/home/usivaraman/CV/P4/VO/', 'filtered_MH05.csv')
with open(gt_file, 'r') as f:
    reader = csv.reader(f)
    next(reader) # skip header
    
    for i,row in enumerate(reader):
        # row =reader[i]
        timestamp1 = np.array(int(row[0]))      
        pos1 = np.array([float(row[1]), float(row[2]), float(row[3])])
        q1 = np.array([float(row[4]), float(row[5]), float(row[6]), float(row[7])])

       
        try:
            # do something with the next item
             rowplus = next(reader)
        except StopIteration:
            # handle the case where there are no more items
            continue
        timestamp2 = np.array(int(rowplus[0]))      
        pos2 = np.array([float(rowplus[1]), float(rowplus[2]), float(rowplus[3])])
        q2 = np.array([float(rowplus[4]), float(rowplus[5]), float(rowplus[6]), float(rowplus[7])])

        relpos= relative_position(pos1, pos2)
        q_diff = quaternion_difference(q1, q2)

        roww = [timestamp1,timestamp2, relpos[0], relpos[1], relpos[2], q_diff[0], q_diff[1], q_diff[2], q_diff[3]]
        print("roww :",roww)
        data.append(roww)
        i = i+1
                

# create a header for filtered csv file
header = ['#timestamp1','#timestamp2', 'dp_RS_R_x [m]', 'dp_RS_R_y [m]', 'dp_RS_R_z [m]', 'dq_RS_w []', 'dq_RS_x []', 'dq_RS_y []', 'dq_RS_z []']



with open('/home/usivaraman/CV/P4/VO/MH05_gt.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(data)
