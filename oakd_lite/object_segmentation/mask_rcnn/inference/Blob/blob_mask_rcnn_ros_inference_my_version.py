from pickletools import uint8
import cv2
import time
import depthai as dai
import numpy as np
from depthai_sdk import PipelineManager, NNetManager, PreviewManager, Previews, FPSHandler, toTensorResult, frameNorm

# ros dependency
import rospy
import cv_bridge
from sensor_msgs.msg import Image, CameraInfo

class mask_rcnn_segmentation():
    def __init__(self):
        # ros config
        self.bridge = cv_bridge.CvBridge()
        rospy.init_node('mask_rcnn_segmentation_blob')
        self.segmentation_result_pub = rospy.Publisher("/oakd_lite/segmentation/image",Image,queue_size=1)
        self.loop_rate = rospy.Rate(30)

        # oakd_lite config
        self.NN_WIDTH = 300
        self.NN_HEIGHT = 300
        NN_PATH = "./weight/maskrcnn_resnet50_300_300_pi_openvino_2021.4_6shave.blob"

        # Create pipeline
        self.pipeline = dai.Pipeline()
        self.pipeline.setOpenVINOVersion(version = dai.OpenVINO.VERSION_2021_4)

        # Define a neural network
        detection_nn = self.pipeline.create(dai.node.NeuralNetwork)
        detection_nn.setBlobPath(str(NN_PATH))
        detection_nn.setNumPoolFrames(4)
        detection_nn.input.setBlocking(False)
        detection_nn.setNumInferenceThreads(2)
        self.threshold = 0.5
        self.region_threshold = 0.5
        self.label_map = ["pi"]
        self.colors = np.random.random(size=(256, 3)) * 256

        # Define rgb camera
        camRgb = self.pipeline.create(dai.node.ColorCamera)
        camRgb.setPreviewSize(self.NN_WIDTH, self.NN_HEIGHT)
        camRgb.setInterleaved(False)
        camRgb.setFps(30)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

        # Define left right camera to create depth
        monoLeft = self.pipeline.create(dai.node.MonoCamera)
        monoRight = self.pipeline.create(dai.node.MonoCamera)
        stereo = self.pipeline.create(dai.node.StereoDepth)
        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P) # THE_400_P, THE_480_P
        monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
        monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        self.depth_width = monoLeft.getResolutionWidth()
        self.depth_height = monoLeft.getResolutionHeight()
        # Align depth map to the perspective of RGB camera, on which inference is done
        stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        stereo.setOutputSize(monoLeft.getResolutionWidth(), monoLeft.getResolutionHeight())
        #stereo.setOutputSize(self.NN_WIDTH,self.NN_HEIGHT)

        # Create outputs
        xoutRgb = self.pipeline.create(dai.node.XLinkOut)
        xoutRgb.setStreamName("rgb")
        xoutNN = self.pipeline.create(dai.node.XLinkOut)
        xoutNN.setStreamName("nn")
        xoutDepth = self.pipeline.create(dai.node.XLinkOut)
        xoutDepth.setStreamName("depth")

        # Linking
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)
        camRgb.preview.link(detection_nn.input)
        stereo.depth.link(xoutDepth.input)
        detection_nn.passthrough.link(xoutRgb.input)
        detection_nn.out.link(xoutNN.input)

    def toTensorResult(self,packet):
        data = {}
        for tensor in packet.getRaw().tensors:
            if tensor.dataType == dai.TensorInfo.DataType.INT:
                data[tensor.name] = np.array(packet.getLayerInt32(tensor.name)).reshape(tensor.dims)
            elif tensor.dataType == dai.TensorInfo.DataType.FP16:
                data[tensor.name] = np.array(packet.getLayerFp16(tensor.name)).reshape(tensor.dims)
            elif tensor.dataType == dai.TensorInfo.DataType.I8:
                data[tensor.name] = np.array(packet.getLayerUInt8(tensor.name)).reshape(tensor.dims)
            else:
                print("Unsupported tensor layer type: {}".format(tensor.dataType))
        return data

    def transformation(self,depth_frame,center_x, center_y):

        distance_z = depth_frame[center_y,center_x]
        expected_3d_center_distance = distance_z

        expected_3d_center_x = ((center_x - self.rgb_u)*distance_z)/self.rgb_fx
        expected_3d_center_y = ((center_y - self.rgb_v)*distance_z)/self.rgb_fy

        return expected_3d_center_distance,expected_3d_center_x,expected_3d_center_y

    def show_boxes_and_regions(self,rgb_frame,depth_frame, boxes, masks):
        for i, box in enumerate(boxes):
            if box[0] == -1:
                break

            cls = int(box[1])
            prob = box[2]

            if prob < self.threshold:
                continue

            bbox = frameNorm(rgb_frame, box[-4:])
            cv2.rectangle(rgb_frame, (bbox[0], bbox[1]-15), (bbox[2], bbox[1]), self.colors[cls], -1)
            cv2.rectangle(rgb_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), self.colors[cls], 1)
            cv2.putText(rgb_frame, f"{self.label_map[cls-1]}: {prob:.2f}", (bbox[0] + 5, bbox[1] - 5), cv2.FONT_HERSHEY_DUPLEX, 0.3, (0, 0, 0), 2)
            cv2.putText(rgb_frame, f"{self.label_map[cls-1]}: {prob:.2f}", (bbox[0] + 5, bbox[1] - 5), cv2.FONT_HERSHEY_DUPLEX, 0.3, (255, 255, 255), 1)

            bbox_w = bbox[2] - bbox[0]
            bbox_h = bbox[3] - bbox[1]

            mask = cv2.resize(masks[i, cls], (bbox_w, bbox_h))
            mask = mask > self.region_threshold
            roi = rgb_frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            roi[mask] = roi[mask] * 0.6 + self.colors[cls] * 0.4
            rgb_frame[bbox[1]:bbox[3], bbox[0]:bbox[2]] = roi
            mask = np.asarray(mask,dtype=np.uint8)
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for i in contours:
                M = cv2.moments(i)
                if M['m00'] != 0:
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    rgb_frame = cv2.circle(rgb_frame, (int(cx+bbox[0]),int(cy+bbox[1])), 7, (0, 0, 255), -1)
                    real_world_z,real_world_x,real_world_y = self.transformation(depth_frame,int(cx),int(cy))
                    position = "{},{},{}".format(round(real_world_x,4), round(real_world_y,4), round(float(real_world_z),4))
                    rgb_frame = cv2.putText(rgb_frame, position, (int(cx),int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return rgb_frame

    def main(self):
        with dai.Device(self.pipeline, usb2Mode = True) as device:
            previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
            nnQueue = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
            
            # Get rgb camera intrinsic
            calibData = device.readCalibration()
            rgb_intrinsic = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.RGB, self.NN_WIDTH, self.NN_HEIGHT))
            self.rgb_fx = rgb_intrinsic[0][0]
            self.rgb_fy = rgb_intrinsic[1][1]
            self.rgb_u = rgb_intrinsic[0][2]
            self.rgb_v = rgb_intrinsic[1][2]

            while not rospy.is_shutdown():
                start = time.time()
                inPreview = previewQueue.get()
                depth = depthQueue.get()
                nnResult = nnQueue.get()

                rgb_frame = inPreview.getCvFrame()
                depthFrame = depth.getFrame() # depthFrame values are in millimeters
                
                output = toTensorResult(nnResult)
                boxes = output["DetectionOutput_647"].squeeze()
                masks = output["Sigmoid_733"]

                rgb_frame = self.show_boxes_and_regions(rgb_frame,depthFrame/1000.0,boxes,masks) # convert from mm to m
                fps  = (1.0/(time.time()-start))
                rgb_frame = cv2.putText(rgb_frame, "fps: %.2f"%(fps), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Mask RCNN result", rgb_frame)
                cv2.imshow("depth result", depthFrame)
                cv2.waitKey(1)

if __name__ == "__main__":
    mask_rcnn = mask_rcnn_segmentation()
    mask_rcnn.main()