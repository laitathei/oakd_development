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

Go to https://blobconverter.luxonis.com/ and convert onnx model to openvino format with `--mean_values=data[123.675,116.28,103.53] --scale_values=data[58.395,57.12,57.375]`

![image](https://github.com/laitathei/oakd_development/blob/master/oakd_lite/image/blobconverter_page.png)
![image](https://github.com/laitathei/oakd_development/blob/master/oakd_lite/image/blobconverter_parameters.png)
