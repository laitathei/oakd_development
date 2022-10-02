## Pre-requisites

Install ros dependency
```
sudo apt-get install ros-noetic-rtabmap
sudo apt-get install ros-noetic-rtabmap-ros
sudo apt-get install ros-noetic-imu-tools
```

Install python requirements
```
python3 -m pip install -r requirements.txt
```

## Mapping with official package

Rgbd odometry
```
python3 oakd_node.py
roslaunch rtabmap_ros rtabmap.launch args:="--delete_db_on_start" depth_topic:=/oakd_lite/depth/image rgb_topic:=/oakd_lite/rgb/image camera_info_topic:=/oakd_lite/rgb/camera_info frame_id:=oakd_frame approx_sync:=true rviz:=true
```

Stereo odometry
```
python3 oakd_node.py
roslaunch rtabmap_ros rtabmap.launch args:="--delete_db_on_start" stereo:=true left_image_topic:=/oakd_lite/left/image_rect right_image_topic:=/oakd_lite/right/image_rect left_camera_info_topic:=/oakd_lite/left/camera_info right_camera_info_topic:=/oakd_lite/right/camera_info frame_id:=oakd_frame approx_sync:=true rviz:=true
```

ICP odometry
```
python3 oakd_node.py
rosrun nodelet nodelet standalone rtabmap_ros/point_cloud_xyz _approx_sync:=true /depth/image:=/oakd_lite/depth/image /depth/camera_info:=/oakd_lite/depth/camera_info _decimation:=4
roslaunch rtabmap_ros rtabmap.launch rtabmap_args:="--delete_db_on_start --Icp/VoxelSize 0.05 --Icp/PointToPlaneRadius 0 --Icp/PointToPlaneK 20 --Icp/CorrespondenceRatio 0.2 --Icp/PMOutlierRatio 0.65 --Icp/Epsilon 0.005 --Icp/PointToPlaneMinComplexity 0 --Odom/ScanKeyFrameThr 0.7 --OdomF2M/ScanMaxSize 15000 --Optimizer/GravitySigma 0.3 --RGBD/ProximityPathMaxNeighbors 1 --Reg/Strategy 1" icp_odometry:=true scan_cloud_topic:=/cloud subscribe_scan_cloud:=true depth_topic:=/oakd_lite/depth/image rgb_topic:=/oakd_lite/rgb/image camera_info_topic:=/oakd_lite/rgb/camera_info approx_sync:=true rviz:=true frame_id:=oakd_frame
```

## Mapping with oakd_development package
Rgbd odometry
```
roslaunch oakd_development oakd_lite_mapping_rgbd_odometry.launch
```

Stereo odometry
```
roslaunch oakd_development oakd_lite_mapping_stereo_odometry.launch
```

ICP odometry
```
roslaunch oakd_development oakd_lite_mapping_icp_odometry.launch
```
