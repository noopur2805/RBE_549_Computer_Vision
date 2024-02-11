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

def create_model_6d_quat(window_size=200):
    input_gyro_acc = Input((window_size, 6))
    lstm1 = Bidirectional(LSTM(128, return_sequences=True))(input_gyro_acc)    
    drop1 = Dropout(0.25)(lstm1)
    lstm2 = Bidirectional(LSTM(128))(drop1)    
    drop2 = Dropout(0.25)(lstm2)    
    output_delta_p = Dense(3)(drop2)
    output_delta_q = Dense(4)(drop2)

    model = Model(inputs = input_gyro_acc, outputs = [output_delta_p, output_delta_q])
    model.summary()
#     model.compile(optimizer = Adam(0.0001), loss = 'mean_squared_error')
#     model.compile(optimizer = Adam(0.0001), loss = ['mean_absolute_error', quaternion_mean_multiplicative_error])
    model.compile(optimizer = Adam(0.0001), loss = ['mean_absolute_error', quaternion_phi_4_error])
    
    return model

window_size = 200
stride = 10

x_gyro = []
x_acc = []

y_delta_p = []
y_delta_q = []

imu_data_filenames = []
gt_data_filenames = []

imu_data_filenames.append('/home/usivaraman/CV/P4/IO/MH_02_easy/mav0/imu0/data.csv')
imu_data_filenames.append('/home/usivaraman/CV/P4/IO/MH_03_medium/mav0/imu0/data.csv')
imu_data_filenames.append('/home/usivaraman/CV/P4/IO/MH_04_difficult/mav0/imu0/data.csv')
imu_data_filenames.append('/home/usivaraman/CV/P4/IO/MH_05_difficult/mav0/imu0/data.csv')


gt_data_filenames.append('/home/usivaraman/CV/P4/IO/MH_02_easy/mav0/state_groundtruth_estimate0/data.csv')
gt_data_filenames.append('/home/usivaraman/CV/P4/IO/MH_03_medium/mav0/state_groundtruth_estimate0/data.csv')
gt_data_filenames.append('/home/usivaraman/CV/P4/IO/MH_04_difficult/mav0/state_groundtruth_estimate0/data.csv')
gt_data_filenames.append('/home/usivaraman/CV/P4/IO/MH_05_difficult/mav0/state_groundtruth_estimate0/data.csv')


for i, (cur_imu_data_filename, cur_gt_data_filename) in enumerate(zip(imu_data_filenames, gt_data_filenames)):
        
        cur_gyro_data, cur_acc_data, cur_pos_data, cur_ori_data = load_euroc_mav_dataset(cur_imu_data_filename, cur_gt_data_filename)
        # print("cur_gyro :",cur_gyro_data)
        [cur_x_gyro, cur_x_acc], [cur_y_delta_p, cur_y_delta_q], init_p, init_q = load_dataset_6d_quat(cur_gyro_data, cur_acc_data, cur_pos_data, cur_ori_data, window_size, stride)
        # print("cur_x_gyro :",cur_x_gyro)
        x_gyro.append(cur_x_gyro)
        x_acc.append(cur_x_acc)

        y_delta_p.append(cur_y_delta_p)
        y_delta_q.append(cur_y_delta_q)

x_gyro = np.vstack(x_gyro)
x_acc = np.vstack(x_acc)

y_delta_p = np.vstack(y_delta_p)
y_delta_q = np.vstack(y_delta_q)

x_gyro, x_acc, y_delta_p, y_delta_q = shuffle(x_gyro, x_acc, y_delta_p, y_delta_q)

input_gyro_acc = np.concatenate((x_gyro,x_acc), axis=-1)



model= create_model_6d_quat(window_size)


model_checkpoint = ModelCheckpoint('model_checkpoint.hdf5', monitor='train_loss', verbose=1)

print("input tensor shape :",input_gyro_acc.shape)
print("output tensor shape :",y_delta_p.shape)
print("output tensor shape :",y_delta_q.shape)

# Train the model
history = model.fit([input_gyro_acc],[y_delta_p, y_delta_q], epochs=100, batch_size=32, verbose=1, callbacks=[model_checkpoint])

# [yhat_delta_p, yhat_delta_q] = model.predict(input_gyro_acc, batch_size=1, verbose=0)
plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
plt.show()
plt.savefig('Plot.png')
model.save('trained_model_IO.h5')
# gt_trajectory = generate_trajectory_6d_quat(init_p, init_q, y_delta_p, y_delta_q)
# pred_trajectory = generate_trajectory_6d_quat(init_p, init_q, yhat_delta_p, yhat_delta_q)
# trajectory_rmse = np.sqrt(np.mean(np.square(np.linalg.norm(pred_trajectory - gt_trajectory, axis=-1))))

# print('Trajectory RMSE, sequence %s: %f' % (cur_imu_data_filename, trajectory_rmse))




