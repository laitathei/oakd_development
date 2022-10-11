import utils
import transforms as T
from engine import train_one_epoch, evaluate
import sys
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
#from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
import torchvision.transforms as transforms
import os
import torch
import numpy as np
import torch.utils.data
import PIL
from PIL import Image, ImageColor, ImageDraw, ImageFont
import time
import cv2
import warnings
from types import FunctionType
from typing import Any, BinaryIO, List, Optional, Tuple, Union

# ros package
import rospy
import rospy
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
import cv_bridge

class segmentation():
    def __init__(self):
        # ros config
        self.bridge = cv_bridge.CvBridge()
        rospy.init_node('mask_rcnn_inference')
        self.segmentation_result_pub = rospy.Publisher("/segmentation_result", Image, queue_size=1)
        rospy.Subscriber("/oakd_lite/rgb/image",Image,self.color_callback)
        rospy.Subscriber("/oakd_lite/depth/image",Image,self.depth_callback)
        rospy.Subscriber("/oakd_lite/rgb/camera_info",CameraInfo,self.color_camera_info_callback)
        self.SavePath = "./weight/model_5.pth"
        self.loop_rate = rospy.Rate(30)
        # split the dataset in train and test set
        torch.manual_seed(1)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.class_names = ["__background__", "grass"]
        self.colors = np.random.randint(0, 255, size=(len(self.class_names), 3))
        self.colors = [tuple(color) for color in self.colors]
        self.num_classes = len(self.class_names)
        self.confident_level = 0.5
        self.model = torch.load(self.SavePath)
        self.model.to(self.device)
        self.model.eval() # put the model in evaluation mode

    def depth_callback(self,data):
        # Depth image callback
        self.depth_image = self.bridge.imgmsg_to_cv2(data)
        self.depth_image = np.array(self.depth_image, dtype=np.float32)
        self.depth_array = self.depth_image/1000.0

    def color_callback(self,data):
        # RGB image callback
        self.rgb_image = self.bridge.imgmsg_to_cv2(data)

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

    def draw_bounding_boxes(self,
        image: torch.Tensor,
        boxes: torch.Tensor,
        labels: Optional[List[str]] = None,
        colors: Optional[Union[List[Union[str, Tuple[int, int, int]]], str, Tuple[int, int, int]]] = None,
        fill: Optional[bool] = False,
        width: int = 1,
        font: Optional[str] = None,
        font_size: Optional[int] = None,) -> torch.Tensor:
    
        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            self._log_api_usage_once(self.draw_bounding_boxes)
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Tensor expected, got {type(image)}")
        elif image.dtype != torch.uint8:
            raise ValueError(f"Tensor uint8 expected, got {image.dtype}")
        elif image.dim() != 3:
            raise ValueError("Pass individual images, not batches")
        elif image.size(0) not in {1, 3}:
            raise ValueError("Only grayscale and RGB images are supported")
        elif (boxes[:, 0] > boxes[:, 2]).any() or (boxes[:, 1] > boxes[:, 3]).any():
            raise ValueError(
                "Boxes need to be in (xmin, ymin, xmax, ymax) format. Use torchvision.ops.box_convert to convert them"
            )

        num_boxes = boxes.shape[0]

        if num_boxes == 0:
            warnings.warn("boxes doesn't contain any box. No box was drawn")
            return image

        if labels is None:
            labels: Union[List[str], List[None]] = [None] * num_boxes  # type: ignore[no-redef]
        elif len(labels) != num_boxes:
            raise ValueError(
                f"Number of boxes ({num_boxes}) and labels ({len(labels)}) mismatch. Please specify labels for each box."
            )

        if colors is None:
            colors = self._generate_color_palette(num_boxes)
        elif isinstance(colors, list):
            if len(colors) < num_boxes:
                raise ValueError(f"Number of colors ({len(colors)}) is less than number of boxes ({num_boxes}). ")
        else:  # colors specifies a single color for all boxes
            colors = [colors] * num_boxes

        colors = [(ImageColor.getrgb(color) if isinstance(color, str) else color) for color in colors]

        if font is None:
            if font_size is not None:
                warnings.warn("Argument 'font_size' will be ignored since 'font' is not set.")
            txt_font = ImageFont.load_default()
        else:
            txt_font = ImageFont.truetype(font=font, size=font_size or 10)

        # Handle Grayscale images
        if image.size(0) == 1:
            image = torch.tile(image, (3, 1, 1))

        print(type(image))
        ndarr = image.permute(1, 2, 0).cpu().numpy()
        print(type(ndarr))
        img_to_draw = PIL.Image.fromarray(ndarr)
        img_boxes = boxes.to(torch.int64).tolist()

        if fill:
            draw = ImageDraw.Draw(img_to_draw, "RGBA")
        else:
            draw = ImageDraw.Draw(img_to_draw)

        for bbox, color, label in zip(img_boxes, colors, labels):  # type: ignore[arg-type]
            if fill:
                fill_color = color + (100,)
                draw.rectangle(bbox, width=width, outline=color, fill=fill_color)
            else:
                draw.rectangle(bbox, width=width, outline=color)

            if label is not None:
                margin = width + 1
                draw.text((bbox[0] + margin, bbox[1] + margin), label, fill=color, font=txt_font)

        return torch.from_numpy(np.array(img_to_draw)).permute(2, 0, 1).to(dtype=torch.uint8)

    def draw_segmentation_masks(self,
        image: torch.Tensor,
        masks: torch.Tensor,
        alpha: float = 0.8,
        colors: Optional[Union[List[Union[str, Tuple[int, int, int]]], str, Tuple[int, int, int]]] = None,
    ) -> torch.Tensor:

        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            self._log_api_usage_once(self.draw_segmentation_masks)
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"The image must be a tensor, got {type(image)}")
        elif image.dtype != torch.uint8:
            raise ValueError(f"The image dtype must be uint8, got {image.dtype}")
        elif image.dim() != 3:
            raise ValueError("Pass individual images, not batches")
        elif image.size()[0] != 3:
            raise ValueError("Pass an RGB image. Other Image formats are not supported")
        if masks.ndim == 2:
            masks = masks[None, :, :]
        if masks.ndim != 3:
            raise ValueError("masks must be of shape (H, W) or (batch_size, H, W)")
        if masks.dtype != torch.bool:
            raise ValueError(f"The masks must be of dtype bool. Got {masks.dtype}")
        if masks.shape[-2:] != image.shape[-2:]:
            raise ValueError("The image and the masks must have the same height and width")

        num_masks = masks.size()[0]
        if colors is not None and num_masks > len(colors):
            raise ValueError(f"There are more masks ({num_masks}) than colors ({len(colors)})")

        if num_masks == 0:
            warnings.warn("masks doesn't contain any mask. No mask was drawn")
            return image

        if colors is None:
            colors = self._generate_color_palette(num_masks)

        if not isinstance(colors, list):
            colors = [colors]
        if not isinstance(colors[0], (tuple, str)):
            raise ValueError("colors must be a tuple or a string, or a list thereof")
        if isinstance(colors[0], tuple) and len(colors[0]) != 3:
            raise ValueError("It seems that you passed a tuple of colors instead of a list of colors")

        out_dtype = torch.uint8

        colors_ = []
        for color in colors:
            if isinstance(color, str):
                color = ImageColor.getrgb(color)
            colors_.append(torch.tensor(color, dtype=out_dtype))

        img_to_draw = image.detach().clone()
        # TODO: There might be a way to vectorize this
        for mask, color in zip(masks, colors_):
            img_to_draw[:, mask] = color[:, None]

        out = image * (1 - alpha) + img_to_draw * alpha
        return out.to(out_dtype)

    def _generate_color_palette(self,num_objects: int):
        palette = torch.tensor([2**25 - 1, 2**15 - 1, 2**21 - 1])
        return [tuple((i * palette) % 255) for i in range(num_objects)]

    def _log_api_usage_once(self,obj: Any) -> None:
        module = obj.__module__
        if not module.startswith("torchvision"):
            module = f"torchvision.internal.{module}"
        name = obj.__class__.__name__
        if isinstance(obj, FunctionType):
            name = obj.__name__
        torch._C._log_api_usage_once(f"{module}.{name}")

    def get_instance_segmentation_model(self,num_classes):
        # load an instance segmentation model pre-trained on COCO
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
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

    def get_transformed_image(self,image):
        image_transposed = np.transpose(image, [2, 0, 1])
        # Convert to uint8 tensor.
        uint8_tensor = torch.tensor(image_transposed, dtype=torch.uint8)
        # Convert to float32 tensor.
        transform = transforms.Compose([transforms.ToTensor(),])
        float32_tensor = transform(image)
        float32_tensor = torch.unsqueeze(float32_tensor, 0)
        return uint8_tensor, float32_tensor

    def filter_detections(self, outputs, dataset_class_names, detection_threshold=0.8):
        #print("detection_threshold: {}".format(detection_threshold))
        pred_scores = outputs[0]['scores'].detach().cpu().numpy()
        pred_classes = [dataset_class_names[i] for i in outputs[0]['labels'].cpu().numpy()]
        pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
        boxes = pred_bboxes[pred_scores >= detection_threshold].astype(np.int32)
        # if len(boxes)==2:
        #     print("_______________")
        #     print(len(boxes))
        #     print(boxes)
        #     print(pred_scores)
        pred_classes = pred_classes[:len(boxes)]
        labels = outputs[0]['labels'][:len(boxes)]
        return boxes, pred_classes, labels, pred_scores

    def draw_boxes(self, boxes, unint8_tensor, pred_classes, labels, colors, fill=False,is_instance=False):
        if is_instance:
            plot_colors = colors=np.random.randint(0, 255, size=(len(boxes), 3))
            plot_colors = [tuple(color) for color in plot_colors]
        else:
            plot_colors = [colors[label] for label in labels]
        if len(boxes)==0:
            plot_colors = self.colors
            no_detection_result = True
            print("No box!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        else:
            no_detection_result = False
        result_with_boxes = self.draw_bounding_boxes(image=unint8_tensor, boxes=torch.tensor(boxes), width=2, colors=plot_colors,labels=pred_classes,fill=fill,font="arial.ttf",font_size=40)
        return result_with_boxes, plot_colors, no_detection_result

    def draw_instance_mask(self, outputs, uint8_tensor, colors, no_detection_result, detection_threshold=0.8):
        # Get all the scores.
        pred_scores = list(outputs[0]['scores'].detach().cpu().numpy())
        # Index of those scores which are above a certain threshold.
        thresholded_preds_inidices = [pred_scores.index(i) for i in pred_scores if i > detection_threshold]
        thresholded_preds_count = len(thresholded_preds_inidices)
        masks = outputs[0]['masks']
        final_masks = masks > 0.5
        final_masks = final_masks.squeeze(1)
        # Discard masks for objects which are below threshold.
        final_masks = final_masks[:thresholded_preds_count]
        seg_result = self.draw_segmentation_masks(uint8_tensor, final_masks,colors=colors,alpha=0.8)
        return seg_result, final_masks

    def transformation(self,center_x, center_y):

        distance_z = self.depth_array[center_y,center_x]
        expected_3d_center_distance = distance_z

        expected_3d_center_x = ((center_x - self.rgb_u)*distance_z)/self.rgb_fx
        expected_3d_center_y = ((center_y - self.rgb_v)*distance_z)/self.rgb_fy

        return expected_3d_center_distance,expected_3d_center_x,expected_3d_center_y

    def main(self):
        while not rospy.is_shutdown():
            rospy.wait_for_message("/oakd_lite/rgb/image",Image)
            rospy.wait_for_message("/oakd_lite/depth/image",Image)
            rospy.wait_for_message("/oakd_lite/rgb/camera_info",CameraInfo)
            start = time.time()
            #img = cv2.imread("./images/train2017/frame0.jpg")
            #self.rgb_image = T.Compose(T.ToTensor(self.rgb_image))
            uint8_frame, float32_frame = self.get_transformed_image(self.rgb_image)
            float32_frame = torch.squeeze(float32_frame)
            with torch.no_grad():
                self.prediction = self.model([float32_frame.to(self.device)])
            float32_frame = float32_frame.mul(255).permute(1, 2, 0).byte().numpy()
            float32_frame = cv2.cvtColor(float32_frame,cv2.COLOR_BGR2RGB)
            # Get the filetered boxes, class names, and label indices.
            boxes, pred_classes, labels, pred_scores = self.filter_detections(self.prediction, self.class_names, self.confident_level)
            print(pred_scores)
            print(pred_classes)
            # Draw boxes and show current frame on screen.
            result, plot_colors, no_detection_result = self.draw_boxes(boxes, uint8_frame, pred_classes, labels, self.colors, is_instance=True)
            # Draw the segmentation map.
            result,final_masks = self.draw_instance_mask(self.prediction, result, plot_colors, no_detection_result, self.confident_level)
            result = np.transpose(result, (1, 2, 0))
            result = np.ascontiguousarray(result, dtype=np.uint8)
            curr_fps = 1.0 / (time.time() - start)
            result = cv2.putText(result, "FPS: %.2f"%(curr_fps), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            for i in range (len(boxes)):
                result = cv2.putText(result, str(round(pred_scores[i],4)), (boxes[i][0], boxes[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            for i in range (len(final_masks)):
                mask_3_channel = np.zeros_like(result)
                mask_3_channel[:,:,0] = final_masks[i].mul(255).byte().cpu().numpy()
                mask_3_channel[:,:,1] = final_masks[i].mul(255).byte().cpu().numpy()
                mask_3_channel[:,:,2] = final_masks[i].mul(255).byte().cpu().numpy()
                # Get the mask center point x y
                contours, hierarchy = cv2.findContours(final_masks[i].mul(255).byte().cpu().numpy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                for i in contours:
                    M = cv2.moments(i)
                    if M['m00'] != 0:
                        cx = int(M['m10']/M['m00'])
                        cy = int(M['m01']/M['m00'])
                    result = cv2.circle(result, (int(cx),int(cy)), 7, (0, 0, 255), -1)
                    real_world_z,real_world_x,real_world_y = self.transformation(int(cx),int(cy))
                    position = "{},{},{}".format(round(real_world_x,4), round(real_world_y,4), round(float(real_world_z),4))
                    result = cv2.putText(result, position, (int(cx-40),int(cy-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # cur_cut = cv2.bitwise_and(float32_frame,mask_3_channel)
            # cv2.imshow("original image",float32_frame)
            # cv2.imshow("mask image",mask_3_channel)
            # cv2.imshow("mix",cur_cut)
            cv2.imshow("color mask",result)
            self.segmentation_result_pub.publish(self.bridge.cv2_to_imgmsg(cv2.cvtColor(result, cv2.COLOR_RGB2BGR))) # Convert from BGR to RGB color format.
            cv2.waitKey(1)
            self.loop_rate.sleep()

if __name__ == '__main__':
    mask_rcnn = segmentation()
    mask_rcnn.main()

