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

from DatasetUtils import *

window_size = 200
stride = 10

def quaternion_phi_4_error(y_true, y_pred):
    return 1 - K.abs(K.batch_dot(y_true, K.l2_normalize(y_pred, axis=-1), axes=-1))


def quaternion_log_phi_4_error(y_true, y_pred):
    return K.log(1e-4 + quaternion_phi_4_error(y_true, y_pred))

# model = load_model('/home/usivaraman/CV/P4/IO/model_checkpoint.hdf5')
model = load_model('/home/usivaraman/CV/P4/IO/model_checkpoint.hdf5', custom_objects={'quaternion_phi_4_error': quaternion_phi_4_error}, compile=False)

timestamp, gyro_data, acc_data, pos_data, ori_data = load_euroc_mav_dataset('/home/usivaraman/CV/P4/IO/MH_01_easy/mav0/imu0/data.csv','/home/usivaraman/CV/P4/IO/MH_01_easy/mav0/state_groundtruth_estimate0/data.csv')



print("length of timestamp:",len(timestamp))
print("length of pose data :",len(pos_data))
print("length of ori data :",len(ori_data))


[x_gyro, x_acc], [y_delta_p, y_delta_q], init_p, init_q = load_dataset_6d_quat(gyro_data, acc_data, pos_data, ori_data, window_size, stride)

input_gyro_acc = np.concatenate((x_gyro,x_acc), axis=-1)
[yhat_delta_p, yhat_delta_q] = model.predict([input_gyro_acc], batch_size=1, verbose=1)

# # print("gt : ",y_delta_p, y_delta_q)
print("estimated : ",len(yhat_delta_p), len(yhat_delta_q))

gt_trajectory = generate_trajectory_6d_quat(init_p, init_q, y_delta_p, y_delta_q)
pred_trajectory = generate_trajectory_6d_quat(init_p, init_q, yhat_delta_p, yhat_delta_q)


with open('stamped_groundtruth.txt', 'w') as f:
    # Write the timestamp and data to the file
    for i in range(len(timestamp)):
        f.write(" {} {} {} {} {} {} {} {}\n".format(timestamp[i],pos_data[i,0],pos_data[i,1],pos_data[i,1],ori_data[i,0],ori_data[i,1],ori_data[i,2],ori_data[i,3] ))

with open('stamped_traj_estimate.txt', 'w') as f:
    # Write the timestamp and data to the file
    for i in range(len(timestamp)):
        f.write(" {} {} {} {} {} {} {} {}\n".format(timestamp[i],yhat_delta_p[i,0],yhat_delta_p[i,1],yhat_delta_p[i,1],yhat_delta_q[i,0],yhat_delta_q[i,1],yhat_delta_q[i,2],yhat_delta_q[i,3] ))

trajectory_rmse = np.sqrt(np.mean(np.square(np.linalg.norm(pred_trajectory - gt_trajectory, axis=-1))))    




# print("gt_trajectory :",gt_trajectory)
# print("pred_trajectory :",pred_trajectory)
print("trajectory_rmse :",trajectory_rmse)

