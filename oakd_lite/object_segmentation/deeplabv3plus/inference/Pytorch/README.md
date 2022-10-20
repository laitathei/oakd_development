## Export pytorch model to onnx
```
python3 export_onnx.py --ckpt model_best.pth.tar --num_class 6 --input_width 640 --input_height 352
```

## Video or Image inference
```
python3 pytorch_deeplabv3plus_image_video_inference.py --in-path ./source --ckpt ./run/grass/deeplab-mobilenet/model_best.pth.tar --backbone mobilenet --num_class 6
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

