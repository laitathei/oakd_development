### Prerequisite:

<details><summary>[click for detail step]</summary>

```
pip3 install matplotlib
pip3 install tensorboardX
pip3 install pillow
pip3 install tqdm

```

</details>

### Dataset preparation:

<details><summary>[click for detail step]</summary>

```
python3 json_to_dataset.py --input_dataset_dir ./dataset --output_dataset_dir ./dataset --image_name dataset --train_val_ratio 0
python3 voc_train_val_split.py
```

</details>

#### Train:

<details><summary>[click for detail step]</summary>

```
python3 train.py --backbone mobilenet --lr 0.007 --workers 1 --epochs 50 --batch-size 8 --gpu-ids 0 --checkname deeplab-mobilenet --dataset grass
```

</details>

#### Reference:

<details><summary>[click for detail step]</summary>

```
https://blog.csdn.net/yx868yx/article/details/113778713
https://github.com/jfzhang95/pytorch-deeplab-xception
https://github.com/luxonis/depthai-experiments/tree/master/gen2-deeplabv3_multiclass#gen2-deeplabv3-multiclass-on-depthai
```

</details>

python3 train.py --backbone mobilenet --lr 0.007 --workers 1 --epochs 50 --batch-size 8 --gpu-ids 0 --checkname deeplab-mobilenet --dataset grass
python3 video2rosbag.py ./source/test.mp4 test.bag
python3 pytorch_deeplabv3plus_image_video_inference.py --in-path ./source --ckpt ./run/grass/deeplab-mobilenet/model_best.pth.tar --backbone mobilenet
python3 pytorch_deeplabv3plus_ros_inference.py
python3 onnxruntime_deeplabv3plus_image_video_inference.py --in-path ./source --weight ./models/model_352_640.onnx
python3 onnxruntime_deeplabv3plus_ros_inference.py
python3 export_onnx.py
python3 blob_deeplabv3plus_realtime_inference.py --nn_model ./models/model_352_640.blob --h 640 --w 352
python3 blob_deeplabv3plus_ros_inference.py

https://netron.app/

https://blog.csdn.net/yx868yx/article/details/113778713
https://github.com/jfzhang95/pytorch-deeplab-xception
