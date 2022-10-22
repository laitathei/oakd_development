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
# download human dataset from https://www.kaggle.com/datasets/bijoyroy/human-segmentation-dataset
python3 coco2labelme.py ./custom_dataset/*.json ./custom_dataset
python3 json_to_dataset.py --input_dataset_dir ./custom_dataset --output_dataset_dir ./custom_dataset
python3 voc_train_val_split.py --jsonfilepath ./custom_dataset/json --saveBasePath ./custom_dataset/ImageSets/Segmentation
```

</details>

#### Train:

<details><summary>[click for detail step]</summary>

  + Modify `mypath.py` line 14
  ```
  return os.getcwd()+'/custom_dataset' # change to your custom dataset path if you want
  ```
  
  + Modify `/dataloaders/datasets/custom_dataset.py` line 14 and 18
  ```
  NUM_CLASSES = 2 # change to your custom dataset class number
  
  base_dir=Path.db_root_dir('custom_dataset'), # change to your custom dataset path if you want
  ```
  
  + Modify `/dataloaders/utils.py` line 31 and 107
  ```
  n_classes = 2 # change to your custom dataset class number
  
  return np.asarray([[0,0,0],[128,0,0]]) # change with your desired mask color for your custom dataset if you want
  ```
  
  + Start training
  ```
  python3 train.py --backbone mobilenet --lr 0.007 --workers 1 --epochs 50 --batch-size 8 --gpu-ids 0 --checkname deeplab-mobilenet --dataset custom_dataset --num_class 6
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
