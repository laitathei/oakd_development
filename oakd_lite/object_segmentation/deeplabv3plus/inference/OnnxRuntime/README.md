## Video or Image inference
```
python3 onnxruntime_deeplabv3plus_image_video_inference.py --in-path ./source --weight ./models/model_352_640.onnx
```

## Convert video to rosbag
```
python3 video2rosbag.py ./source/test.mp4 test.bag
```

## ROS inference
```
rosbag play test.bag
roslaunch oakd_node oakd_node.launch
python3 onnxruntime_deeplabv3plus_ros_inference.py
```

