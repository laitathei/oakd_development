## Export pytorch model to onnx
```
python3 export_onnx.py
```

## Video or Image inference
```
python3 pytorch_deeplabv3plus_image_video_inference.py --in-path ./source --ckpt ./run/grass/deeplab-mobilenet/model_best.pth.tar --backbone mobilenet
```

## Convert video to rosbag
```
python3 video2rosbag.py ./source/test.mp4 test.bag
```

## ROS inference
```
rosbag play test.bag
roslaunch oakd_node oakd_node.launch
python3 pytorch_deeplabv3plus_ros_inference.py
```

