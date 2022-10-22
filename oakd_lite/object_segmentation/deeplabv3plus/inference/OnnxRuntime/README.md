## Video or Image inference
```
python3 onnxruntime_deeplabv3plus_image_video_inference.py --in-path ./source --weight ./models/model_352_640.onnx --num_class 6
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

## Export to Blob

Go to https://blobconverter.luxonis.com/ and convert onnx model to openvino format with `mean_values=[127.5,127.5,127.5] --scale_values=[255,255,255]`

![image](https://github.com/laitathei/oakd_development/blob/master/oakd_lite/image/blobconverter_page.png)
![image](https://github.com/laitathei/oakd_development/blob/master/oakd_lite/image/blobconverter_parameters.png)
