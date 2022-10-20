import numpy as np
import onnxruntime
import cv2
import time

import cv_bridge
import rospy
from sensor_msgs.msg import Image, CameraInfo, PointCloud2

class deeplabv3plus():
    def __init__(self):
        self.bridge = cv_bridge.CvBridge()
        rospy.init_node('deeplabv3plus_onxxruntime')

        # Subscribe color and depth image
        rospy.Subscriber("/oakd_lite/rgb/image",Image,self.color_callback)
        rospy.Subscriber("/oakd_lite/depth/image",Image,self.depth_callback)
        # Subscribe camera info
        rospy.Subscriber("/oakd_lite/rgb/camera_info",CameraInfo,self.color_camera_info_callback)
        rospy.Subscriber("/oakd_lite/depth/camera_info",CameraInfo,self.depth_camera_info_callback)
        # Publish segmentation image
        self.segmentation_result_pub = rospy.Publisher("/oakd_lite/segmentation/image",Image,queue_size=1)        

        # model config
        self.weight = "./models/model_352_640.onnx" # model_height_width.onnx

        self.sess = onnxruntime.InferenceSession(self.weight)
        self.input_name = self.sess.get_inputs()[0].name
        self.label_name = self.sess.get_outputs()[0].name
        self.dataset = "custom_dataset"
        self.num_class = 6
        
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

    def decode_seg_map_sequence(self,label_masks, n_classes, dataset='grass'): # change
        rgb_masks = []
        for label_mask in label_masks:
            rgb_mask = self.decode_segmap(label_mask, dataset, n_classes)
            rgb_masks.append(rgb_mask)
        rgb_masks = np.array(rgb_masks).transpose([0, 3, 1, 2])
        return rgb_masks

    def decode_segmap(self,label_mask, dataset, n_classes, plot=False):
        """Decode segmentation class labels into a color image
        Args:
            label_mask (np.ndarray): an (M,N) array of integer values denoting
            the class label at each spatial location.
            plot (bool, optional): whether to show the resulting color image
            in a figure.
        Returns:
            (np.ndarray, optional): the resulting decoded color image.
        """
        if dataset == 'custom_dataset':
            n_classes = n_classes
            label_colours = self.get_custom_dataset_labels()
        else:
            raise NotImplementedError

        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        for ll in range(0, n_classes):
            r[label_mask == ll] = label_colours[ll, 0]
            g[label_mask == ll] = label_colours[ll, 1]
            b[label_mask == ll] = label_colours[ll, 2]
        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def get_custom_dataset_labels(self):
        return np.asarray([[0,0,0],[128,0,0],[0,128,0],[0,0,128],[256,0,0],[0,256,0],[0,0,256]]) # change with your desired mask color for your custom dataset if you want

    def composed_transforms(self,sample,mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        img /= 255.0
        img -= mean
        img /= std
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        return {'image': img,
                'label': mask}

    def main(self):
        while not rospy.is_shutdown():
            rospy.wait_for_message("/oakd_lite/rgb/image",Image)
            #rospy.wait_for_message("/oakd_lite/depth/image",Image)
            s_time = time.time()

            origin_image = cv2.cvtColor(self.rgb_image,cv2.COLOR_BGR2RGB)
            mask_image = cv2.cvtColor(self.rgb_image,cv2.COLOR_BGR2GRAY)
            sample = {'image': origin_image, 'label': mask_image}

            tensor_in = self.composed_transforms(sample)['image']
            tensor_in = np.expand_dims(tensor_in, axis=0) # (1, 3, 352, 640)
            pred_onnx = self.sess.run([self.label_name],{self.input_name: tensor_in.astype(np.float32)})[0] # (1, number of class, 352, 640)
            # np.argmax(pred_onnx[:3],axis=1) return the max value indice in each row
            decode = self.decode_seg_map_sequence(np.argmax(pred_onnx[:3],axis=1),self.num_class,dataset=self.dataset) # (1, number of class, 352, 640)
            grid_image = np.squeeze(decode, axis=0)
            grid_image = grid_image*255 # mul(255)
            grid_image = grid_image+0.5 # add_(0.5)
            grid_image = np.clip(grid_image,0,255) # clamp_(0, 255)
            grid_image_ndarr = np.array(grid_image,dtype=np.uint8)
            grid_image_ndarr = np.transpose(grid_image_ndarr,(1,2,0))
            origin_image = np.array(origin_image,dtype=np.uint8)
            u_time = time.time()
            fps = 1.0 / (u_time-s_time)
            mix = cv2.addWeighted(origin_image,0.8,grid_image_ndarr,1.0,0)
            mix = cv2.putText(mix, "fps: %.2f"%(fps), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("mix",cv2.cvtColor(mix,cv2.COLOR_BGR2RGB))
            cv2.waitKey(1)



if __name__ == "__main__":
    segmentation = deeplabv3plus()
    segmentation.main()

# python3 pytorch_deeplabv3plus_ros_inference.py
