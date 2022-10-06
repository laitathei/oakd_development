# OAKD camera development demo

This repo contains demo application for development purpose

* oakd_lite
    * [Camera Calibration](/oakd_lite/camera_calibration)
        * [Kalibr](/oakd_lite/camera_calibration/kalibr) (Done stereo only)
        * [Image pipeline] (Done)
    * [Visual Inertial Odometry](/oakd_lite/visual_inertial_odometry)
        * [Vins-Mono] (To do)
        * [Vins-Fusion](/oakd_lite/visual_inertial_odometry/VINS-Fusion) (To do)
    * [Visual SLAM](/oakd_lite/visual_slam)
        * [RTAB-map](/oakd_lite/visual_slam/rtabmap) (Done mapping) (rgbd,stereo,icp)
        * ORB-SLAM3 (To do)
    * [Object segmentation](/oakd_lite/object_segmentation)
        * [Yolov7](/oakd_lite/object_segmentation/yolov7)
            * [Inference](/oakd_lite/object_segmentation/yolov7/inference)
                * [OnnxRuntime](/oakd_lite/object_segmentation/yolov7/inference/OnnxRuntime) (Done)
                * [Pytorch](/oakd_lite/object_segmentation/yolov7/inference/Pytorch) (Done)
            * [Train](https://github.com/laitathei/YOLOv7-Pytorch-Segmentation/tree/5ddbb5f1684cbca11779414e4fc2d8be01133379) (Done)
    * [Object recognition](/oakd_lite/object_recognition) (To do)


# Reference
- http://wiki.ros.org/rtabmap_ros
- https://docs.luxonis.com/en/latest/
- https://github.com/WongKinYiu/yolov7
- https://github.com/HKUST-Aerial-Robotics/VINS-Fusion
- https://github.com/ethz-asl/kalibr
- https://support.stereolabs.com/hc/en-us/articles/360012749113-How-can-I-use-Kalibr-with-the-ZED-Mini-camera-in-ROS-
- https://blog.csdn.net/sunqin_csdn/article/details/104874374/
- http://wiki.ros.org/camera_calibration/Tutorials/MonocularCalibration
- https://www.youtube.com/watch?v=2XM2Rb2pfyQ&ab_channel=FirstPrinciplesofComputerVision
- https://github.com/HKUST-Aerial-Robotics/VINS-Fusion
