import cv_bridge
import rospy
from sensor_msgs.msg import Image, CameraInfo, PointCloud2

import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.segment.general import process_mask, scale_masks
from utils.segment.plots import plot_masks
from utils.torch_utils import select_device, smart_inference_mode
from utils.augmentations import letterbox

class yolov7_segmentation():
    def __init__(self):
        self.bridge = cv_bridge.CvBridge()
        rospy.init_node('object_detection_tensorrt')

        # Subscribe color and depth image
        rospy.Subscriber("/oakd_lite/rgb/image",Image,self.color_callback)
        rospy.Subscriber("/oakd_lite/depth/image",Image,self.depth_callback)
        # Subscribe camera info
        rospy.Subscriber("/oakd_lite/rgb/camera_info",CameraInfo,self.color_camera_info_callback)
        rospy.Subscriber("/oakd_lite/depth/camera_info",CameraInfo,self.depth_camera_info_callback)
        # Publish segmentation image
        self.segmentation_result_pub = rospy.Publisher("/oakd_lite/segmentation/image",Image,queue_size=1)

        self.weights = 'weight/best.pt'  # model.pt path(s)
        self.data = 'config/coco.yaml'  # dataset.yaml path
        self.imgsz = [640, 640]  # inference size (height, width)
        self.conf_thres = 0.25  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold
        self.max_det = 1000  # maximum detections per image
        self.classes = None
        self.agnostic_nms = False
        self.line_thickness = 3
        self.hide_labels = False
        self.hide_conf = False
        self.batch_size = 1  # batch_size

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

    @smart_inference_mode()
    def main(self):

        # Load model
        device = select_device("")
        model = DetectMultiBackend(self.weights, device=device, dnn=False, data=self.data, fp16=False)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(self.imgsz, s=stride)  # check image size

        while not rospy.is_shutdown():
            s = ""
            rospy.wait_for_message("/oakd_lite/rgb/image",Image)
            rospy.wait_for_message("/oakd_lite/depth/image",Image)
            start = time.time()

            # Run inference
            model.warmup(imgsz=(1 if pt else self.batch_size, 3, *imgsz))  # warmup
            dt = (Profile(), Profile(), Profile())
            #im0 = cv2.imread(im0) # cv format
            im0 = self.rgb_image
            print("____________________")
            print(im0.shape)
            im = letterbox(im0, self.imgsz, stride=32, auto=True)[0]  # padded resize
            print(im.shape)
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)  # contiguous
            with dt[0]:
                im = torch.from_numpy(im).to(device)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with dt[1]:
                pred, out = model(im, augment=False, visualize=False)
                proto = out[1]

            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det, nm=32)

            # Process predictions
            for i, det in enumerate(pred):  # per image

                annotator = Annotator(im0, line_width=self.line_thickness, example=str(names))
                if len(det):
                    masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC

                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}"  # add to string show the number of class

                    print(s)

                    # Mask plotting ----------------------------------------------------------------------------------------
                    mcolors = [colors(int(cls), True) for cls in det[:, 5]]
                    im_masks = plot_masks(im[i], masks, mcolors)  # image with masks shape(imh,imw,3)
                    annotator.im = scale_masks(im.shape[2:], im_masks, im0.shape)  # scale to original h, w
                    # BBox plotting
                    for *xyxy, conf, cls in reversed(det[:, :6]):
                        c = int(cls)  # integer class
                        label = None if self.hide_labels else (names[c] if self.hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))

                # Stream results
                im0 = annotator.result()
                fps  = (1.0/(time.time()-start))
                im0 = cv2.putText(im0, "fps: %.2f"%(fps), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                self.segmentation_result_pub.publish(self.bridge.cv2_to_imgmsg(im0))
                cv2.namedWindow("Segmentation result", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                cv2.resizeWindow("Segmentation result", im0.shape[1], im0.shape[0])
                cv2.imshow("Segmentation result", im0)
                cv2.waitKey(1)  # 1 millisecond

if __name__ == "__main__":
    segmentation = yolov7_segmentation()
    segmentation.main()