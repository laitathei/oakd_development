### Prerequisite:

<details><summary>[click for detail step]</summary>

  ```
  wget https://download.pytorch.org/whl/cu110/torch-1.7.0%2Bcu110-cp38-cp38-linux_x86_64.whl
  wget https://download.pytorch.org/whl/cu110/torchvision-0.8.1%2Bcu110-cp38-cp38-linux_x86_64.whl
  pip3 install torch-1.7.0+cu110-cp38-cp38-linux_x86_64.whl torchvision-0.8.1+cu110-cp38-cp38-linux_x86_64.whl
  pip3 install blobconverter
  pip3 install labelme==3.16.2
  pip3 install openvino==2021.4.2
  pip3 install openvino-dev==2021.4.2
  ```

</details>

### Dataset preparation:

<details><summary>[click for detail step]</summary>

```
rosbag record /oakd_lite/rgb/image
python3 rosbag2video.py *.bag
python3 extract_frames_opencv.py *.mp4
python3 json_to_dataset.py --input_dataset_dir ./dataset --output_dataset_dir ./dataset --image_name dataset --train_val_ratio 0
```

</details>

#### Train:

<details><summary>[click for detail step]</summary>

```
python3 train.py
```

</details>
