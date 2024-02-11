# Deep Visual Inertial Odometry -https://rbe549.github.io/spring2023/proj/p4/

  
For algorithm details, please refer to:
* Robust Stereo Visual Inertial Odometry for Fast Autonomous Flight, Ke Sun et al. (2017)
* A Multi-State Constraint Kalman Filterfor Vision-aided Inertial Navigation, Anastasios I. Mourikis et al. (2006)  

## Requirements
* Python 3.6+
* numpy
* scipy
* cv2
* tensorflow
* keras

## Dataset
* [EuRoC MAV](http://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets): visual-inertial datasets collected on-board a MAV. The datasets contain stereo images, synchronized IMU measurements, and ground-truth.  
This project implements data loader and data publisher for EuRoC MAV dataset.



Inertial Odometry
## Run  
python ..\usivaraman_P4_ph2\Code\IO\Test_Code.py

Visual Odometry
## Run  
python ..\usivaraman_P4_ph2\Code\VO\test_code.py 
  

