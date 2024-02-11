import tensorflow as tf
import cv2
import numpy as np
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import argparse

from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import load_model
from keras.optimizers import Adam
from tensorflow.keras.layers import Input, Conv2D, Flatten, Lambda, Dense,MaxPooling2D
from sklearn.utils import shuffle
from keras.models import Sequential, Model
from keras.layers import Bidirectional, LSTM, CuDNNLSTM, Dropout, Dense, Input, Layer, Conv1D, MaxPooling1D, concatenate
from keras.initializers import Constant
from keras.optimizers import Adam
from keras.losses import mean_absolute_error
from keras import backend as K
from time import time
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Flatten, Lambda, Dense,MaxPooling2D

import numpy as np
import pandas as pd
import quaternion
import scipy.interpolate
import numpy as np
import quaternion
from keras.utils import Sequence
import tfquaternion as tfq
import tensorflow as tf
from keras import backend as K


def quaternion_phi_3_error(y_true, y_pred):
    return tf.acos(K.abs(K.batch_dot(y_true, K.l2_normalize(y_pred, axis=-1), axes=-1)))


def quaternion_phi_4_error(y_true, y_pred):
    return 1 - K.abs(K.batch_dot(y_true, K.l2_normalize(y_pred, axis=-1), axes=-1))


def quaternion_log_phi_4_error(y_true, y_pred):
    return K.log(1e-4 + quaternion_phi_4_error(y_true, y_pred))

def quat_mult_error(y_true, y_pred):
    q = tfq.Quaternion(y_pred).normalized()
    p = tfq.Quaternion(y_true).normalized()
    error = 2.0 * tf.square(tfq.signed_angle(q * p, tfq.Quaternion(tf.zeros_like(y_pred, dtype=tf.float32))))
    return error


def quaternion_mean_multiplicative_error(y_true, y_pred):
    print("y_true.shape :",y_true.shape)
    print("y_true.shape :",y_pred.shape)
    return tf.reduce_mean(quat_mult_error(y_true, y_pred))

def read_images_gt_function():

    stacked_images_batch =[]
    stacked_pos =[]
    stacked_quat =[]
    # gt_file = os.path.join('/home/usivaraman/CV/P4/VO/', 'filtered_gt.csv')
    gt_data_filenames = []
    img1_data_filenames =[]
    img2_data_filenames =[]

    timestamps =[]
    # img1_data_filenames.append('/home/usivaraman/CV/P4/VO/MH_02_easy/mav0/cam0/data/')
    # img1_data_filenames.append('/home/usivaraman/CV/P4/VO/MH_03_medium/mav0/cam0/data/')
    img1_data_filenames.append('/home/usivaraman/CV/P4/VO/MH_04_difficult/mav0/cam0/data/')
    # img1_data_filenames.append('/home/usivaraman/CV/P4/VO/MH_05_difficult/mav0/cam0/data/')

    # img2_data_filenames.append('/home/usivaraman/CV/P4/VO/MH_02_easy/mav0/cam1/data/')
    # img2_data_filenames.append('/home/usivaraman/CV/P4/VO/MH_03_medium/mav0/cam1/data/')
    img2_data_filenames.append('/home/usivaraman/CV/P4/VO/MH_04_difficult/mav0/cam1/data/')
    # img2_data_filenames.append('/home/usivaraman/CV/P4/VO/MH_05_difficult/mav0/cam1/data/')


    # gt_data_filenames.append('/home/usivaraman/CV/P4/VO/MH02_gt.csv')
    # gt_data_filenames.append('/home/usivaraman/CV/P4/VO/MH03_gt.csv')
    gt_data_filenames.append('/home/usivaraman/CV/P4/VO/MH04_gt.csv')
    # gt_data_filenames.append('/home/usivaraman/CV/P4/VO/MH05_gt.csv')
    
    for i, (cam0,cam1, gt_file) in enumerate(zip(img1_data_filenames,img2_data_filenames, gt_data_filenames)):
        with open(gt_file, 'r') as f:
            reader = csv.reader(f)
            next(reader) # skip header
            
            for i,row in enumerate(reader):

                ts1 =row[0]
                ts2 =row[1]
                timestamps.append(ts1)

                ts1_name = ts1 +'.png'
                ts2_name = ts2 +'.png'

                
                left_image_path1 = os.path.join(cam0,ts1_name)
                right_image_path1 = os.path.join(cam1,ts1_name)

                left_image_path2 = os.path.join(cam0,ts2_name)
                right_image_path2 = os.path.join(cam1,ts2_name)


                # print("left_image_path1 :",left_image_path1)
                # print("right_image_path1 :",right_image_path1)
                # print("left_image_path2 :",left_image_path2)
                # print("right_image_path2 :",right_image_path2)

                # Read left images and stack 
                left_image_t = cv2.imread(left_image_path1)
                left_image_t1 = cv2.imread(left_image_path2)

                # print("left_image_t :",np.array(left_image_t).shape)
                # print("left_image_t1 :",np.array(left_image_t1).shape)

                stacked_left_images = np.concatenate((left_image_t,left_image_t1), axis=-1)
                
            
                # print("shape of left stacked_imaged_batch :",np.array(stacked_left_images).shape)

                # Read right images and stack
                right_image_t = cv2.imread(right_image_path1)
                right_image_t1 = cv2.imread(right_image_path2)

                # print("right_image_t :",np.array(right_image_t).shape)
                # print("right_image_t1 :",np.array(right_image_t1).shape)

                stacked_right_images = np.concatenate((right_image_t,right_image_t1), axis=-1)
                # print("shape of right stacked_imaged_batch :",np.array(stacked_right_images).shape)


                stacked_images = np.concatenate((stacked_left_images,stacked_right_images), axis=-1)
                stacked_images_batch.append(stacked_images)

                # print("shape of stacked_imaged_batch :",np.array(stacked_images_batch).shape)

                dpos = np.array([float(row[2]), float(row[3]), float(row[4])])
                dquat = np.array([float(row[5]), float(row[6]), float(row[7]), float(row[8])])

                stacked_pos.append(dpos)
                stacked_quat.append(dquat)

    # stacked_images_batch = tf.convert_to_tensor(stacked_images_batch, dtype=tf.float32)
    # stacked_pos = tf.convert_to_tensor(stacked_pos, dtype=tf.float32)
    # stacked_quat = tf.convert_to_tensor(stacked_quat, dtype=tf.float32)

    # # Create a TensorFlow dataset
    # dataset = tf.data.Dataset.from_tensor_slices((stacked_images_batch, stacked_pos, stacked_quat))

    # # Batch the dataset
    # dataset = dataset.batch(batch_size)

    return timestamps,stacked_images_batch,stacked_pos,stacked_quat

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

def Model():
    height = 480
    width = 752
    num_channels = 12
    input_shape = (height, width, num_channels)
    inputs = Input(shape=input_shape)
    num_filters = 32  # number of filters in each convolutional layer
    kernel_size = (3, 3)  # size of the convolutional kernel

    # inputs = Input(shape=input_shape)
    print("inputs.shape : ",inputs.shape)
    # Add a set of 2D convolutional layers with stride of 2 and pooling layers
    x = Conv2D(filters=num_filters, kernel_size=kernel_size, strides=(2,2), padding='same', activation='relu')(inputs)
    print("x.shape Conv2D :",x.shape)
    x = MaxPooling2D(pool_size=(2,2))(x)
    print("x.shape after maxpooling2d:",x.shape)
    x = Conv2D(filters=num_filters, kernel_size=kernel_size, strides=(2,2), padding='same', activation='relu')(x)
    print("x.shape Conv2D:",x.shape)
    x = MaxPooling2D(pool_size=(2,2))(x)
    print("x.shape after maxpooling2D:",x.shape)
    # Flatten output tensor and add a dense layer
    x = Flatten()(x)
    print("x.shape after flatten:",x.shape)
    x = Dense(units=128, activation='relu')(x)

    # Add another dense layer to reshape output to 7x1
    x = Dense(units=7, activation='relu')(x)

    # Flatten the output of the TCN layer
    x = Flatten()(x)

    # Add two fully connected layers with 3x1 and 4x1 neurons respectively
    pos = Dense(units=3, activation='relu')(x)
    orientation = Dense(units=4, activation='relu')(x)

    # Define model
    model = tf.keras.models.Model(inputs=inputs, outputs=[pos, orientation])
    model.summary()
    return model



timestamps,stacked_images_batch,stacked_pos,stacked_quat = read_images_gt_function()

# stacked_images_batch = np.vstack(stacked_images_batch)
stacked_pos = np.vstack(stacked_pos)
stacked_quat = np.vstack(stacked_quat)

print("input tensor shape :",np.array(stacked_images_batch).shape)
print("output tensor shape :",stacked_pos.shape)
print("output tensor shape :",stacked_quat.shape)

model = load_model('/home/usivaraman/CV/P4/VO/trained_VO_model.h5', custom_objects={'quaternion_phi_4_error': quaternion_phi_4_error}, compile=False)
[pred_pos,pred_quat] = model.predict([np.array(stacked_images_batch)], batch_size=1, verbose=1)

print("pred_pos",pred_pos)
print("pred_quat",pred_quat)

with open('stamped_groundtruth.txt', 'w') as f:
    # Write the timestamp and data to the file
    for i in range(len(timestamps)):
        f.write(" {} {} {} {} {} {} {} {}\n".format(timestamps[i],stacked_pos[i,0],stacked_pos[i,1],stacked_pos[i,1],stacked_quat[i,0],stacked_quat[i,1],stacked_quat[i,2],stacked_quat[i,3] ))

with open('stamped_traj_estimate.txt', 'w') as f:
    # Write the timestamp and data to the file
    for i in range(len(timestamps)):
        f.write(" {} {} {} {} {} {} {} {}\n".format(timestamps[i],pred_pos[i,0],pred_pos[i,1],pred_pos[i,1],pred_quat[i,0],pred_quat[i,1],pred_quat[i,2],pred_quat[i,3] ))