## Video or Image inference
```
python3 onnxruntime_yolov7_segmentation_image_video_inference.py --weights ./weight/best.onnx --source ./source/test.jpg
```

## ROS inference
```
roscore
python3 oakd_node.py
python3 onnxruntime_yolov7_segmentation_ros_inference.py
```

## Visualize ONNX graph
- https://netron.app/

