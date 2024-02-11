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


def read_images_gt_function(batch_size):

    stacked_images_batch =[]
    stacked_pos =[]
    stacked_quat =[]
    gt_file = os.path.join('/home/usivaraman/CV/P4/VO/', 'filtered_gt.csv')
    with open(gt_file, 'r') as f:
        reader = csv.reader(f)
        next(reader) # skip header
        
        for i,row in enumerate(reader):

            ts1 =row[0]
            ts2 =row[1]

            ts1_name = ts1 +'.png'
            ts2_name = ts2 +'.png'

            
            left_image_path1 = os.path.join('/home/usivaraman/CV/P4/VO/mav0/cam0/data',ts1_name)
            right_image_path1 = os.path.join('/home/usivaraman/CV/P4/VO/mav0/cam1/data',ts1_name)

            left_image_path2 = os.path.join('/home/usivaraman/CV/P4/VO/mav0/cam0/data',ts2_name)
            right_image_path2 = os.path.join('/home/usivaraman/CV/P4/VO/mav0/cam1/data',ts2_name)


            # print("left_image_path1 :",left_image_path1)
            # print("right_image_path1 :",right_image_path1)
            # print("left_image_path2 :",left_image_path2)
            # print("right_image_path2 :",right_image_path2)

            # Read left images and stack 
            left_image_t = cv2.imread(left_image_path1)
            left_image_t1 = cv2.imread(left_image_path2)
            stacked_left_images = np.hstack((left_image_t, left_image_t1))

            # Read right images and stack
            right_image_t = cv2.imread(right_image_path1)
            right_image_t1 = cv2.imread(right_image_path2)
            stacked_right_images = np.hstack((right_image_t, right_image_t1))


            stacked_images = np.hstack((stacked_left_images, stacked_right_images))
            stacked_images_batch.append(stacked_images)

            dpos = np.array([float(row[2]), float(row[3]), float(row[4])])
            dquat = np.array([float(row[5]), float(row[6]), float(row[7]), float(row[8])])

            stacked_pos.append(dpos)
            stacked_quat.append(dquat)

    return np.asarray(stacked_images_batch),np.asarray(stacked_pos),np.asarray(stacked_quat)

stacked_images_batch,stacked_pos,stacked_quat = read_images_gt_function()

print("stacked_images_batch :",stacked_images_batch)
print("stacked_pos :",stacked_pos)
print("stacked_quat :",stacked_quat)
