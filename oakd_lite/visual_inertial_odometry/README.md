- Install dependency
```
sudo apt-get install cmake
sudo apt-get install libgoogle-glog-dev libgflags-dev
sudo apt-get install libatlas-base-dev
sudo apt-get install libeigen3-dev
sudo apt-get install libsuitesparse-dev
```

- Download [latest stable release Ceres Solver](http://ceres-solver.org/installation.html)
```
wget http://ceres-solver.org/ceres-solver-2.1.0.tar.gz
tar zxf ceres-solver-2.1.0.tar.gz
mkdir ceres-bin
cd ceres-bin
cmake ../ceres-solver-2.1.0
make -j3
make test
sudo make install
```

- Build Vins-Fusion with ROS
```
cd ~/catkin_ws/src
git clone https://github.com/HKUST-Aerial-Robotics/VINS-Fusion.git
cd ../
catkin_make
source ~/catkin_ws/devel/setup.bash
```

- Modify Vins-Fusion CMakeList
```
#set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_STANDARD 14)
```

- Add header file to Chessboard.h
```
#include <opencv2/imgproc/types_c.h>
#include <opencv2/calib3d/calib3d_c.h>
```

- Add header file to CameraCalibration.h
```
#include <opencv2/imgproc/types_c.h>
#include <opencv2/imgproc/imgproc_c.h>
```

- Add header file to BRIEF.h
```
#include <opencv2/imgproc/types_c.h>
```

- Add header file to feature_tracker.h
```
#include <opencv2/highgui.hpp>
#include <opencv2/cvconfig.h>
#include <opencv2/imgproc/types_c.h>
```

- Add header file to pose_graph.h and keyframe.h
```
#include <opencv2/imgproc/imgproc_c.h>
```

- Modify KITTIGPSTest.cpp and KITTIOdomTest.cpp
```
CV_LOAD_IMAGE_GRAYSCALE -> cv::IMREAD_GRAYSCALE
```

- Kalibr result explain
```
                  [r11 r12 r13 Tx]   [R T]
T_cn_cnm1 (4x4) = [r21 r22 r23 Ty] = [0 1]
                  [r31 r32 r33 Tz]
                  [0 0 0 1]
R (3x3) is come from baseline q vector (quaternion vector) [q1 q2 q3 q4], you need to convert it to rotation matrix
T (3x1) is come from baseline t vector (translation vector) [Tx Ty Tz] or [t1 t2 t3]
```

- Turn quaternion to rotation matrix
```
# change q1,q2,q3,q4 before you use
python3 quaternion2rotation.py
```

- Place result to vins-fusion config.yaml
```
body_T_cam0: !!opencv-matrix  # Inverse of Kalibr result, (transpose for rotation matrix, T'=-R'T)
   rows: 4
   cols: 4
   dt: d
   data: [0.9987499,0.00259418,-0.0499191,-0.09603513,
          -0.00291667,0.99997534,-0.0063884,-0.00264404,
          0.04990129,0.00652601,0.99873283,0.02017904,
          0, 0, 0, 1]

body_T_cam1: !!opencv-matrix # Inverse of Kalibr result, (transpose for rotation matrix, T'=-R'T)
   rows: 4
   cols: 4
   dt: d
   data: [0.9987499,-0.00291667,0.04990129,-0.09603513,
          0.00259418,0.99997534,0.00652601,-0.00264404,
          -0.0499191,-0.0063884,0.99873283,0.02017904,
          0, 0, 0, 1]
```

- oakd_lite example
```
cd ~/catkin_ws/src/oakd_development/oakd_lite/visual_inertial_odometry/vins_fusion
roslaunch vins vins_rviz.launch
python3 oakd_node.py
rosrun vins vins_node ~/catkin_ws/src/oakd_development/oakd_lite/visual_inertial_odometry/vins_fusion/config/oakd_lite_stereo.yaml 
```
