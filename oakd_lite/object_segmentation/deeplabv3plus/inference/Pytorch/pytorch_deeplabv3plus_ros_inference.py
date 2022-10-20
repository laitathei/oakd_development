import argparse
import os
import numpy as np
import time
import cv2

import cv_bridge
import rospy
from sensor_msgs.msg import Image, CameraInfo, PointCloud2

from modeling.deeplab import *
from dataloaders import custom_transforms as tr
from torchvision import transforms
from dataloaders.utils import  *
from torchvision.utils import make_grid, save_image

class deeplabv3plus():
    def __init__(self):
        self.bridge = cv_bridge.CvBridge()
        rospy.init_node('deeplabv3plus_pytorch')

        # Subscribe color and depth image
        rospy.Subscriber("/oakd_lite/rgb/image",Image,self.color_callback)
        rospy.Subscriber("/oakd_lite/depth/image",Image,self.depth_callback)
        # Subscribe camera info
        rospy.Subscriber("/oakd_lite/rgb/camera_info",CameraInfo,self.color_camera_info_callback)
        rospy.Subscriber("/oakd_lite/depth/camera_info",CameraInfo,self.depth_camera_info_callback)
        # Publish segmentation image
        self.segmentation_result_pub = rospy.Publisher("/oakd_lite/segmentation/image",Image,queue_size=1)

        # network config
        self.num_classes = 2
        self.crop_size = 513
        self.ckpt='./run/grass/deeplab-mobilenet/model_best.pth.tar'
        self.freeze_bn=False
        self.sync_bn=False
        self.out_stride=16
        self.backbone = "mobilenet"
        self.dataset = "custom_dataset"
        self.model = DeepLab(num_classes=self.num_classes,
                        backbone=self.backbone,
                        output_stride=self.out_stride,
                        sync_bn=self.sync_bn,
                        freeze_bn=self.freeze_bn)
        ckpt = torch.load(self.ckpt, map_location='cpu')
        self.model.load_state_dict(ckpt['state_dict'])
        self.model = self.model.cuda()
        self.model.eval()
        self.composed_transforms = transforms.Compose([
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

    def color_callback(self,data):
        # RGB image callback
        self.rgb_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

    def depth_callback(self,data):
        # Depth image callback
        self.depth_image = self.bridge.imgmsg_to_cv2(data)
        self.depth_image = np.array(self.depth_image, dtype=np.float32)
        self.depth_array = self.depth_image/1000.0

    # this is depth camera info callback
    def depth_camera_info_callback(self, data):
        self.depth_height = data.height
        self.depth_width = data.width

        # Pixels coordinates of the principal point (center of projection)
        self.depth_u = data.P[2]
        self.depth_v = data.P[6]

        # Focal Length of image (multiple of pixel width and height)
        self.depth_fx = data.P[0]
        self.depth_fy = data.P[5]

    # this is color camera info callback
    def color_camera_info_callback(self, data):
        self.rgb_height = data.height
        self.rgb_width = data.width

        # Pixels coordinates of the principal point (center of projection)
        self.rgb_u = data.P[2]
        self.rgb_v = data.P[6]

        # Focal Length of image (multiple of pixel width and height)
        self.rgb_fx = data.P[0]
        self.rgb_fy = data.P[5]

    def main(self):
        while not rospy.is_shutdown():
            rospy.wait_for_message("/oakd_lite/rgb/image",Image)
            #rospy.wait_for_message("/oakd_lite/depth/image",Image)
            s_time = time.time()

            image = cv2.cvtColor(self.rgb_image,cv2.COLOR_BGR2RGB)
            target = cv2.cvtColor(self.rgb_image,cv2.COLOR_BGR2GRAY)
            sample = {'image': image, 'label': target}
            tensor_in = self.composed_transforms(sample)['image'].unsqueeze(0)
            tensor_in = tensor_in.cuda()
            with torch.no_grad():
                output = self.model(tensor_in)

            grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy(),dataset=self.dataset), 3, normalize=False, range=(0, 255))
            grid_image_ndarr = grid_image.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            image = np.array(image,dtype=np.uint8)
            mix = cv2.addWeighted(image,0.8,grid_image_ndarr,1.0,0)
            u_time = time.time()
            fps = 1.0 / (u_time-s_time)
            #image = cv2.putText(image, "fps: %.2f"%(fps), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            #grid_image_ndarr = cv2.putText(grid_image_ndarr, "fps: %.2f"%(fps), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            mix = cv2.putText(mix, "fps: %.2f"%(fps), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            #cv2.imshow("mask",cv2.cvtColor(grid_image_ndarr,cv2.COLOR_BGR2RGB))
            #cv2.imshow("original",cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
            cv2.imshow("mix",cv2.cvtColor(mix,cv2.COLOR_BGR2RGB))
            cv2.waitKey(1)

if __name__ == "__main__":
    segmentation = deeplabv3plus()
    segmentation.main()

# python3 pytorch_deeplabv3plus_ros_inference.py
