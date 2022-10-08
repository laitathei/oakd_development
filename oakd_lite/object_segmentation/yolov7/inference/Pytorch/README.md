## Export pytorch model to onnx
```
python3 export.py --weights ./weight/best.pt --include onnx --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 384 640 --simplify
```

## Video or Image inference
```
python3 pytorch_yolov7_segmentation_image_video_inference.py --weights ./weight/best.pt --source ./source/test.jpg
```

## ROS inference
```
roslaunch oakd_node oakd_node.launch
python3 pytorch_yolov7_segmentation_ros_inference.py
```
