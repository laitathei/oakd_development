### Kalibr:

<details><summary>[click to see]</summary>
+ Install ROS dependency
```
sudo apt-get install ros-noetic-vision-opencv
sudo apt-get install ros-noetic-image-transport-plugins
sudo apt-get install ros-noetic-cmake-modules
```

+ Install dependency
```
sudo apt-get install python3-setuptools
sudo apt-get install python3-rosinstall
sudo apt-get install ipython
sudo apt-get install libeigen3-dev
sudo apt-get install libboost-all-dev
sudo apt-get install doxygen
sudo apt-get install libopencv-dev
sudo apt-get install python3-software-properties
sudo apt-get install software-properties-common
sudo apt-get install libpoco-dev
sudo apt-get install python3-matplotlib
sudo apt-get install python3-numpy
sudo apt-get install python-numpy
sudo apt-get install python3-scipy
sudo apt-get install python3-git
sudo apt-get install python3-pip
sudo apt-get install python3-pyx
sudo apt-get install libtbb-dev
sudo apt-get install libblas-dev
sudo apt-get install liblapack-dev
sudo apt-get install python3-catkin-tools
sudo apt-get install libv4l-dev
pip3 install python-igraph --upgrade
pip3 install pyx
pip3 install attrdict
pip3 install -U wxPython # it will wait a long time
```

+ Build kalibr with ROS
```
cd ~/catkin_ws/src
git clone https://github.com/ethz-asl/kalibr.git
cd ../
catkin_make
source ~/catkin_ws/devel/setup.bash
```

+ Create own aprilgrid
```
cd ~/catkin_ws/src/oakd_development/oakd_lite/camera_calibration/kalibr/aslam_offline_calibration/kalibr/python
python3 kalibr_create_target_pdf --type apriltag --nx [column_number] --ny [row_number] --tsize [target_width_size] --tspace [target_spacing_percent]
python3 kalibr_create_target_pdf --type apriltag --nx 6 --ny 6 --tsize 0.022 --tspace 0.3
```

+ Create own checkerboard
```
cd ~/catkin_ws/src/oakd_development/oakd_lite/camera_calibration/kalibr/aslam_offline_calibration/kalibr/python
python3 kalibr_create_target_pdf --type checkerboard --nx [column_number] --ny [row_number] --tsize [target_width_size] --tspace [target_spacing_percent]
python3 kalibr_create_target_pdf --type checkerboard --nx 8 --ny 6 --csx 0.025 --csy 0.025
```

+ Prepare ROS bag

```
roscore
python3 oakd_node.py
rosbag record /oakd_lite/left/image_rect /oakd_lite/right/image_rect
```

+ Get the camera parameters
```
python3 kalibr_calibrate_cameras --bag ./left_right.bag --topics /oakd_lite/left/image_rect /oakd_lite/right/image_rect --models pinhole-radtan pinhole-radtan --target ./april.yaml
```

---

</details>

### Camera calibration (Image pipeline):

<details><summary>[click to see]</summary>
+ Install ROS dependency
```
sudo apt install ros-noetic-image-pipeline
sudo apt install ros-noetic-camera-calibration
```

+ Mono camera calibration
```
roscore
python3 oakd_node.py
rosrun camera_calibration cameracalibrator.py --size 8x6 --square 0.025 image:=/oakd_lite/rgb/image camera:=/oakd_lite/rgb/camera_info --no-service-check
```

---

</details>

