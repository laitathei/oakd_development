## Export pytorch model to onnx
```
python3 export_onnx.py
```

## Video or Image inference
```
python3 pytorch_mask_rcnn_segmentation_image_video_inference.py --in-path ./source --weight ./weight/model_5.pth
```

## ROS inference
```
rosbag play test.bag
roslaunch oakd_node oakd_node.launch
python3 pytorch_mask_rcnn_segmentation_ros_inference.py
```

