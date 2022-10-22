import argparse
import os
import numpy as np
import time
import cv2

from modeling.deeplab import *
from dataloaders import custom_transforms as tr
from PIL import Image
from torchvision import transforms
from dataloaders.utils import  *
from torchvision.utils import make_grid

def main():

    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Inferencing")
    parser.add_argument('--in-path', type=str, required=True, help='image to test')
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--ckpt', type=str, default='deeplab-resnet.pth',
                        help='saved model')
    parser.add_argument('--num_class', type=int, help='number of class')
    args = parser.parse_args()

    num_classes = args.num_class
    crop_size = 513
    ckpt='./run/grass/deeplab-mobilenet/model_best.pth.tar'
    freeze_bn=False
    sync_bn=False
    out_stride=16

    model_s_time = time.time()
    model = DeepLab(num_classes=num_classes,
                    backbone=args.backbone,
                    output_stride=out_stride,
                    sync_bn=sync_bn,
                    freeze_bn=freeze_bn)
    dataset = "custom_dataset"
    ckpt = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])
    model = model.cuda()
    model_u_time = time.time()
    model_load_time = model_u_time-model_s_time
    model.eval()
    print("model load time is {}".format(model_load_time))

    composed_transforms = transforms.Compose([
        tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        tr.ToTensor()])
    #print(os.listdir(args.in_path))
    for name in os.listdir(args.in_path):
        if name.endswith('.jpg'):
            s_time = time.time()
            image = cv2.imread(args.in_path+"/"+name)
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            target = cv2.imread(args.in_path+"/"+name)
            target = cv2.cvtColor(target,cv2.COLOR_BGR2GRAY)
            sample = {'image': image, 'label': target}
            tensor_in = composed_transforms(sample)['image'].unsqueeze(0)
            tensor_in = tensor_in.cuda()
            #print(tensor_in.shape) # torch.Size([1, 3, 352, 640])
            with torch.no_grad():
                output = model(tensor_in)
            grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy(),dataset=dataset), 3, normalize=False, range=(0, 255))
            grid_image_ndarr = grid_image.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            mix = cv2.addWeighted(image,0.8,grid_image_ndarr,1.0,0)
            #cv2.imshow("mask",cv2.cvtColor(grid_image_ndarr,cv2.COLOR_BGR2RGB))
            #cv2.imshow("original",cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
            cv2.imshow("mix",cv2.cvtColor(mix,cv2.COLOR_BGR2RGB))
            cv2.waitKey(0)
            u_time = time.time()
            img_time = u_time-s_time
            print("image:{} time: {} ".format(name,img_time))
            #print("image save in in_path.")
        elif name.endswith('.mp4'):
            cap = cv2.VideoCapture(args.in_path+"/"+name)
            while cap.isOpened():
                s_time = time.time()
                ret, frame = cap.read()
                # if frame is read correctly ret is True
                if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break
                image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                target = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                sample = {'image': image, 'label': target}
                tensor_in = composed_transforms(sample)['image'].unsqueeze(0)

                tensor_in = tensor_in.cuda()
                with torch.no_grad():
                    output = model(tensor_in)

                grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy(),dataset=dataset), 3, normalize=False, range=(0, 255))
                grid_image_ndarr = grid_image.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                image = np.array(image,dtype=np.uint8)
                mix = cv2.addWeighted(image,0.8,grid_image_ndarr,1.0,0)
                u_time = time.time()
                fps = 1.0 / (u_time-s_time)
                image = cv2.putText(image, "fps: %.2f"%(fps), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                grid_image_ndarr = cv2.putText(grid_image_ndarr, "fps: %.2f"%(fps), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                mix = cv2.putText(mix, "fps: %.2f"%(fps), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                #cv2.imshow("mask",cv2.cvtColor(grid_image_ndarr,cv2.COLOR_BGR2RGB))
                #cv2.imshow("original",cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
                cv2.imshow("mix",cv2.cvtColor(mix,cv2.COLOR_BGR2RGB))
                cv2.waitKey(1)

if __name__ == "__main__":
   main()


# python3 pytorch_deeplabv3plus_image_video_inference.py --in-path ./source --ckpt ./run/grass/deeplab-mobilenet/model_best.pth.tar --backbone mobilenet --num_class 6
