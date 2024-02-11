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

def quat_mult_error(y_true, y_pred):
    q = tfq.Quaternion(y_pred).normalized()
    p = tfq.Quaternion(y_true).normalized()
    error = 2.0 * tf.square(tfq.signed_angle(q * p, tfq.Quaternion(tf.zeros_like(y_pred, dtype=tf.float32))))
    return error


def quaternion_mean_multiplicative_error(y_true, y_pred):
    print("y_true.shape :",y_true.shape)
    print("y_true.shape :",y_pred.shape)
    return tf.reduce_mean(quat_mult_error(y_true, y_pred))
def interpolate_3dvector_linear(input, input_timestamp, output_timestamp):
    assert input.shape[0] == input_timestamp.shape[0]
    func = scipy.interpolate.interp1d(input_timestamp, input, axis=0)
    interpolated = func(output_timestamp)
    return interpolated
def load_euroc_mav_dataset(imu_data_filename, gt_data_filename):
    gt_data = pd.read_csv(gt_data_filename).values    
    imu_data = pd.read_csv(imu_data_filename).values

    # print("gt_data :",gt_data[:,0])
    # print("imu_data:",imu_data)
    gyro_data = interpolate_3dvector_linear(imu_data[:, 1:4], imu_data[:, 0], gt_data[:, 0])
    acc_data = interpolate_3dvector_linear(imu_data[:, 4:7], imu_data[:, 0], gt_data[:, 0])
    pos_data = gt_data[:, 1:4]
    ori_data = gt_data[:, 4:8]

    return  gt_data[:, 0],gyro_data, acc_data, pos_data, ori_data

def force_quaternion_uniqueness(q):

    q_data = quaternion.as_float_array(q)

    if np.absolute(q_data[0]) > 1e-05:
        if q_data[0] < 0:
            return -q
        else:
            return q
    elif np.absolute(q_data[1]) > 1e-05:
        if q_data[1] < 0:
            return -q
        else:
            return q
    elif np.absolute(q_data[2]) > 1e-05:
        if q_data[2] < 0:
            return -q
        else:
            return q
    else:
        if q_data[3] < 0:
            return -q
        else:
            return q

def quaternion_phi_4_error(y_true, y_pred):
    return 1 - K.abs(K.batch_dot(y_true, K.l2_normalize(y_pred, axis=-1), axes=-1))


def quaternion_log_phi_4_error(y_true, y_pred):
    return K.log(1e-4 + quaternion_phi_4_error(y_true, y_pred))

def load_dataset_6d_quat(gyro_data, acc_data, pos_data, ori_data, window_size=200, stride=10):
    #gyro_acc_data = np.concatenate([gyro_data, acc_data], axis=1)

    init_p = pos_data[window_size//2 - stride//2, :]
    init_q = ori_data[window_size//2 - stride//2, :]

    #x = []
    x_gyro = []
    x_acc = []
    y_delta_p = []
    y_delta_q = []

    for idx in range(0, gyro_data.shape[0] - window_size - 1, stride):
        #x.append(gyro_acc_data[idx + 1 : idx + 1 + window_size, :])
        # print("inside")
        x_gyro.append(gyro_data[idx + 1 : idx + 1 + window_size, :])
        x_acc.append(acc_data[idx + 1 : idx + 1 + window_size, :])

        p_a = pos_data[idx + window_size//2 - stride//2, :]
        p_b = pos_data[idx + window_size//2 + stride//2, :]

        q_a = quaternion.from_float_array(ori_data[idx + window_size//2 - stride//2, :])
        q_b = quaternion.from_float_array(ori_data[idx + window_size//2 + stride//2, :])

        delta_p = np.matmul(quaternion.as_rotation_matrix(q_a).T, (p_b.T - p_a.T)).T

        delta_q = force_quaternion_uniqueness(q_a.conjugate() * q_b)

        y_delta_p.append(delta_p)
        y_delta_q.append(quaternion.as_float_array(delta_q))
   
    return [x_gyro, x_acc], [y_delta_p, y_delta_q], init_p, init_q

def generate_trajectory_6d_quat(init_p, init_q, y_delta_p, y_delta_q):
    cur_p = np.array(init_p)
    cur_q = quaternion.from_float_array(init_q)
    pred_p = []
    pred_p.append(np.array(cur_p))

    for [delta_p, delta_q] in zip(y_delta_p, y_delta_q):
        cur_p = cur_p + np.matmul(quaternion.as_rotation_matrix(cur_q), delta_p.T).T
        cur_q = cur_q * quaternion.from_float_array(delta_q).normalized()
        pred_p.append(np.array(cur_p))

    return np.reshape(pred_p, (len(pred_p), 3))