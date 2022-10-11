# from pytorch official vision package
import utils
import transforms as T
from engine import train_one_epoch, evaluate

# from torchvision lib
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

import cv2
import blobconverter
import shutil
import sys
import os
import torch
import numpy as np
import torch.utils.data
from PIL import Image

class custom_dataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "origin_image"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "mask_image"))))
 
    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "origin_image", self.imgs[idx])
        mask_path = os.path.join(self.root, "mask_image", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance with 0 being background
        mask = Image.open(mask_path)
 
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
 
        # split the color-encoded mask into a set of binary masks
        masks = mask == obj_ids[:, None, None]
 
        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        # print((masks+0).dtype)
        masks = torch.as_tensor(masks+0, dtype=torch.uint8)
 
        image_id = torch.tensor([idx])

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
 
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
 
        if self.transforms is not None:
            img, target = self.transforms(img, target)
 
        return img, target
 
    def __len__(self):
        return len(self.imgs)

def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, box_detections_per_img=20, rpn_post_nms_top_n_test=50)
 
    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
 
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
 
    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
 
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,hidden_layer,num_classes)
    
    return model

def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
 
    return T.Compose(transforms)

# use the PennFudan dataset and defined transformations
custom_dataset_train = custom_dataset('./dataset/train', get_transform(train=True))
custom_dataset_test = custom_dataset('./dataset/train', get_transform(train=False))
 
# split the dataset in train and test set
torch.manual_seed(1)
indices = torch.randperm(len(custom_dataset_train)).tolist()
dataset = torch.utils.data.Subset(custom_dataset_train, indices[:-10])
dataset_test = torch.utils.data.Subset(custom_dataset_test, indices[-10:])
 
# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0,collate_fn=utils.collate_fn)
 
data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0,collate_fn=utils.collate_fn)
 
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
 
# the dataset has two classes only - background and person
num_classes = 2
 
# get the model using the helper function
model = get_instance_segmentation_model(num_classes)
# move model to the right device
model.to(device)
 
# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)
 
# the learning rate scheduler decreases the learning rate by 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)
 
# training
num_epochs = 20
for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
 
    # update the learning rate
    lr_scheduler.step()
 
    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)
    
    if (epoch+1) % 5==0:
        model_name = "./output/model_"+str(epoch+1)+".pth"
        model_state_dict_name = "./output/model_state_dict"+str(epoch+1)+".pth"
        torch.save(model, model_name)
        torch.save(model.state_dict(), model_state_dict_name)
        print("save model!!")

device = torch.device('cpu')
model.to(device)
model.eval() # put the model in evaluation mode

# Convert to OpenVINO IR
sys.path.append('./openvino_contrib/modules/mo_pytorch')

import mo_pytorch
mo_pytorch.convert(model, input_shape=[1, 3, 360, 640], model_name='maskrcnn_resnet50_360_640', scale = 255)

xml_file = "./maskrcnn_resnet50_360_640.xml"
bin_file = "./maskrcnn_resnet50_360_640.bin"
blob_path = blobconverter.from_openvino(
    xml=xml_file,
    bin=bin_file,
    data_type="FP16",
    shaves=6,
    version="2021.4"
)

shutil.copy(str(blob_path), "./output")
print("Done export openvino")