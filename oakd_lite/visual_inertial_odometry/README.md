### Vins-Fusion:

<details><summary>[click for detail step]</summary>

  + #### Prerequisite:

    <details><summary>[click for detail step]</summary>

      + Install dependency
      ```
      sudo apt-get install cmake
      sudo apt-get install libgoogle-glog-dev libgflags-dev
      sudo apt-get install libatlas-base-dev
      sudo apt-get install libeigen3-dev
      sudo apt-get install libsuitesparse-dev
      ```

      + Download [latest stable release Ceres Solver](http://ceres-solver.org/installation.html)
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

      + Build Vins-Fusion with ROS
      ```
      cd ~/catkin_ws/src
      git clone https://github.com/HKUST-Aerial-Robotics/VINS-Fusion.git
      cd ../
      catkin_make
      source ~/catkin_ws/devel/setup.bash
      ```

      + Modify Vins-Fusion CMakeList
      ```
      #set(CMAKE_CXX_FLAGS "-std=c++11")
      set(CMAKE_CXX_STANDARD 14)
      ```

      + Add header file to Chessboard.h
      ```
      #include <opencv2/imgproc/types_c.h>
      #include <opencv2/calib3d/calib3d_c.h>
      ```

      + Add header file to CameraCalibration.h
      ```
      #include <opencv2/imgproc/types_c.h>
      #include <opencv2/imgproc/imgproc_c.h>
      ```

      + Add header file to BRIEF.h
      ```
      #include <opencv2/imgproc/types_c.h>
      ```

      + Add header file to feature_tracker.h
      ```
      #include <opencv2/highgui.hpp>
      #include <opencv2/cvconfig.h>
      #include <opencv2/imgproc/types_c.h>
      ```

      + Add header file to pose_graph.h and keyframe.h
      ```
      #include <opencv2/imgproc/imgproc_c.h>
      ```

      + Modify KITTIGPSTest.cpp and KITTIOdomTest.cpp
      ```
      CV_LOAD_IMAGE_GRAYSCALE -> cv::IMREAD_GRAYSCALE
      ```

      + Kalibr result explain
      ```
                        [r11 r12 r13 Tx]   [R T]
      T_cn_cnm1 (4x4) = [r21 r22 r23 Ty] = [0 1]
                        [r31 r32 r33 Tz]
                        [0 0 0 1]
      R (3x3) is come from baseline q vector (quaternion vector) [q1 q2 q3 q4], you need to convert it to rotation matrix
      T (3x1) is come from baseline t vector (translation vector) [Tx Ty Tz] or [t1 t2 t3]
      ```

      + Turn quaternion to rotation matrix
      ```
      # change q1,q2,q3,q4 before you use
      python3 quaternion2rotation.py
      ```

      + Place result to vins-fusion config.yaml
      ```
      body_T_cam0: !!opencv-matrix  # T_ic:(cam0 to imu0): 
        rows: 4
        cols: 4
        dt: d
        data: [-0.0096388,-0.00337655,0.99994784,0.00059751,
                -0.99971708,-0.02171366,-0.00970989,-0.00045322,
                0.02174531,-0.99975853,-0.0031663,0.0000265,
                0, 0, 0, 1]

      body_T_cam1: !!opencv-matrix # T_ic:(cam1 to imu0):
        rows: 4
        cols: 4
        dt: d
        data: [0.01041671,-0.00427108,0.99993662,0.00171211,
                -0.99970228,-0.02210983,0.01031983,-0.07777024,
                0.02206435,-0.99974642,-0.00450012,0.00153162,
                0, 0, 0, 1]
      ```

    </details>

  + #### oakd_lite Vins_Fusion example:

      <details><summary>[click for detail step]</summary>

      + #### Mono:

        <details><summary>[click for detail step]</summary>

        ```
        roslaunch oakd_vins_fusion mono_imu.launch
        rosrun vins vins_node ~/catkin_ws/src/oakd_development/oakd_lite/visual_inertial_odometry/oakd_vins_fusion/config/mono_imu/oakd_lite_mono_imu.yaml 
        ```

        </details>

      + #### Stereo:

        <details><summary>[click for detail step]</summary>

        ```
        roslaunch oakd_vins_fusion stereo.launch
        rosrun vins vins_node ~/catkin_ws/src/oakd_development/oakd_lite/visual_inertial_odometry/oakd_vins_fusion/config/stereo/oakd_lite_stereo.yaml
        ```

        </details>

      + #### Stereo + Imu

        <details><summary>[click for detail step]</summary>

        ```
        roslaunch oakd_vins_fusion stereo_imu.launch
        rosrun vins vins_node ~/catkin_ws/src/oakd_development/oakd_lite/visual_inertial_odometry/oakd_vins_fusion/config/stereo_imu/oakd_lite_stereo_imu.yaml
        ```

        </details>

</details>