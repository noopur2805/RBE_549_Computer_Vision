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



cam0_csv_file = "/home/usivaraman/CV/P4/VO/MH_05_difficult/mav0/cam0/data.csv"
cam1_csv_file = "/home/usivaraman/CV/P4/VO/MH_05_difficult/mav0/cam1/data.csv"

with open(cam0_csv_file, "r") as f:
    reader = csv.reader(f)
    next(reader) # Skip header row

    cam0_data = [(int(row[0]), row[1]) for row in reader]

with open(cam1_csv_file, "r") as f:
    reader = csv.reader(f)
    next(reader) # Skip header row

    cam1_data = [(int(row[0]), row[1]) for row in reader]



# Load timestamps and imagepaths for cam 0
timestamps = []
cam0_paths = []

for cam0_entry in cam0_data:
    
        timestamps.append(cam0_entry[0])
        cam0_paths.append(os.path.join("/home/usivaraman/CV/P4/VO/MH_05_difficult/mav0/cam0/data", cam0_entry[1]))

# read timestamps and values in gt and filter 

data =[]
gt_file = os.path.join('/home/usivaraman/CV/P4/VO/MH_05_difficult/mav0/state_groundtruth_estimate0/', 'data.csv')
with open(gt_file, 'r') as f:
    reader = csv.reader(f)
    next(reader) # skip header
    
    for row in reader:
        timestamp = np.array(int(row[0]))

        if(timestamp in timestamps):

        
            pos = np.array([float(row[1]), float(row[2]), float(row[3])])
            q_RS = np.array([float(row[4]), float(row[5]), float(row[6]), float(row[7])])
            roww = [timestamp, row[1], row[2], row[3], row[4], row[5], row[6], row[7]]
            # print("roww :",roww)
            data.append(roww)
                

# create a header for filtered csv file
header = ['#timestamp', 'p_RS_R_x [m]', 'p_RS_R_y [m]', 'p_RS_R_z [m]', 'q_RS_w []', 'q_RS_x []', 'q_RS_y []', 'q_RS_z []']



with open('/home/usivaraman/CV/P4/VO/filtered_MH05.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(data)