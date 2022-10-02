#!/usr/bin/env python3

import cv2
import numpy as np
import depthai as dai
import sys
from pathlib import Path

# ros package
import rospy
from sensor_msgs.msg import Image, CameraInfo, PointCloud2,RegionOfInterest
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovariance,TwistWithCovariance
import cv_bridge as opencv_cv_bridge
from cv_bridge.boost.cv_bridge_boost import getCvType
import tf

class CvBridgeError(TypeError):
    """
    This is the error raised by :class:`cv_bridge.CvBridge` methods when they fail.
    """
    pass


class CvBridge(object):
    """
    The CvBridge is an object that converts between OpenCV Images and ROS Image messages.

       .. doctest::
           :options: -ELLIPSIS, +NORMALIZE_WHITESPACE

           >>> import cv2
           >>> import numpy as np
           >>> from cv_bridge import CvBridge
           >>> br = CvBridge()
           >>> dtype, n_channels = br.encoding_as_cvtype2('8UC3')
           >>> im = np.ndarray(shape=(480, 640, n_channels), dtype=dtype)
           >>> msg = br.cv2_to_imgmsg(im)  # Convert the image to a message
           >>> im2 = br.imgmsg_to_cv2(msg) # Convert the message to a new image
           >>> cmprsmsg = br.cv2_to_compressed_imgmsg(im)  # Convert the image to a compress message
           >>> im22 = br.compressed_imgmsg_to_cv2(msg) # Convert the compress message to a new image
           >>> cv2.imwrite("this_was_a_message_briefly.png", im2)

    """

    def __init__(self):
        import cv2
        self.cvtype_to_name = {}
        self.cvdepth_to_numpy_depth = {cv2.CV_8U: 'uint8', cv2.CV_8S: 'int8', cv2.CV_16U: 'uint16',
                                       cv2.CV_16S: 'int16', cv2.CV_32S:'int32', cv2.CV_32F:'float32',
                                       cv2.CV_64F: 'float64'}

        for t in ["8U", "8S", "16U", "16S", "32S", "32F", "64F"]:
            for c in [1, 2, 3, 4]:
                nm = "%sC%d" % (t, c)
                self.cvtype_to_name[getattr(cv2, "CV_%s" % nm)] = nm

        self.numpy_type_to_cvtype = {'uint8': '8U', 'int8': '8S', 'uint16': '16U',
                                        'int16': '16S', 'int32': '32S', 'float32': '32F',
                                        'float64': '64F'}
        self.numpy_type_to_cvtype.update(dict((v, k) for (k, v) in self.numpy_type_to_cvtype.items()))

    def dtype_with_channels_to_cvtype2(self, dtype, n_channels):
        return '%sC%d' % (self.numpy_type_to_cvtype[dtype.name], n_channels)

    def encoding_to_cvtype2(self, encoding):
        try:
            return getCvType(encoding)
        except RuntimeError as e:
            raise CvBridgeError(e)

    def cv2_to_imgmsg(self, cvim,frame_id, encoding = "passthrough"):
        """
        Convert an OpenCV :cpp:type:`cv::Mat` type to a ROS sensor_msgs::Image message.

        :param cvim:      An OpenCV :cpp:type:`cv::Mat`
        :param encoding:  The encoding of the image data, one of the following strings:

           * ``"passthrough"``
           * one of the standard strings in sensor_msgs/image_encodings.h

        :rtype:           A sensor_msgs.msg.Image message
        :raises CvBridgeError: when the ``cvim`` has a type that is incompatible with ``encoding``

        If encoding is ``"passthrough"``, then the message has the same encoding as the image's OpenCV type.
        Otherwise desired_encoding must be one of the standard image encodings

        This function returns a sensor_msgs::Image message on success, or raises :exc:`cv_bridge.CvBridgeError` on failure.
        """
        if not isinstance(cvim, (np.ndarray, np.generic)):
            raise TypeError('Your input type is not a numpy array')
        img_msg = Image()
        img_msg.header.stamp = rospy.Time.now() # or rospy.get_rostime()
        img_msg.header.frame_id = frame_id

        img_msg.height = cvim.shape[0]
        img_msg.width = cvim.shape[1]
        if len(cvim.shape) < 3:
            cv_type = self.dtype_with_channels_to_cvtype2(cvim.dtype, 1)
        else:
            cv_type = self.dtype_with_channels_to_cvtype2(cvim.dtype, cvim.shape[2])
        if encoding == "passthrough":
            img_msg.encoding = cv_type
        else:
            img_msg.encoding = encoding
            # Verify that the supplied encoding is compatible with the type of the OpenCV image
            if self.cvtype_to_name[self.encoding_to_cvtype2(encoding)] != cv_type:
                raise CvBridgeError("encoding specified as %s, but image has incompatible type %s" % (encoding, cv_type))
        if cvim.dtype.byteorder == '>':
            img_msg.is_bigendian = True
        img_msg.data = cvim.tostring()
        img_msg.step = len(img_msg.data) // img_msg.height

        return img_msg

class oakd_lite():
    def __init__(self):
        # ros config
        #self.bridge = opencv_cv_bridge.CvBridge()
        self.bridge = CvBridge()
        rospy.init_node('oakd_lite')
        self.rgb_pub = rospy.Publisher("/oakd_lite/rgb/image", Image, queue_size=1)
        self.depth_pub = rospy.Publisher("/oakd_lite/depth/image", Image, queue_size=1)
        self.left_pub = rospy.Publisher("/oakd_lite/left/image_rect", Image, queue_size=1)
        self.right_pub = rospy.Publisher("/oakd_lite/right/image_rect", Image, queue_size=1)
        self.left_info_pub = rospy.Publisher("/oakd_lite/left/camera_info", CameraInfo, queue_size=1)
        self.right_info_pub = rospy.Publisher("/oakd_lite/right/camera_info", CameraInfo, queue_size=1)
        self.rgb_info_pub = rospy.Publisher("/oakd_lite/rgb/camera_info", CameraInfo, queue_size=1)
        self.depth_info_pub = rospy.Publisher("/oakd_lite/depth/camera_info", CameraInfo, queue_size=1)
        self.loop_rate = rospy.Rate(15)
        
        # oakd_lite config
        syncNN = True

        # Create pipeline
        self.pipeline = dai.Pipeline()
        # Define sources and outputs
        camRgb = self.pipeline.create(dai.node.ColorCamera)
        monoLeft = self.pipeline.create(dai.node.MonoCamera)
        monoRight = self.pipeline.create(dai.node.MonoCamera)
        stereo = self.pipeline.create(dai.node.StereoDepth)

        xoutRgb = self.pipeline.create(dai.node.XLinkOut)
        xoutDepth = self.pipeline.create(dai.node.XLinkOut)
        xout_rectif_left  = self.pipeline.create(dai.node.XLinkOut)
        xout_rectif_right = self.pipeline.create(dai.node.XLinkOut)

        xoutRgb.setStreamName("rgb")
        xoutDepth.setStreamName("depth")
        xout_rectif_left.setStreamName('rectified_left')
        xout_rectif_right.setStreamName('rectified_right')

        # Properties
        #camRgb.setPreviewSize(416, 416)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        camRgb.setInterleaved(False)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        camRgb.setIspScale(1, 3) # (1,3)=640x360 origin from 1920x1080
        self.rgb_width, self.rgb_height = camRgb.getIspSize()

        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P) # THE_400_P, THE_480_P
        monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
        monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        # setting node configs
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        self.depth_width = monoLeft.getResolutionWidth()
        self.depth_height = monoLeft.getResolutionHeight()
        # Align depth map to the perspective of RGB camera, on which inference is done
        stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        #stereo.setOutputSize(monoLeft.getResolutionWidth(), monoLeft.getResolutionHeight())

        # Linking
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)
        camRgb.isp.link(xoutRgb.input)
        stereo.depth.link(xoutDepth.input)
        stereo.rectifiedLeft.link(xout_rectif_left.input)
        stereo.rectifiedRight.link(xout_rectif_right.input)

    def left_camera_info(self,D,K,R,P,frame_id):
        roi = RegionOfInterest()
        self.left_info = CameraInfo()
        self.left_info.header.stamp = rospy.Time.now() # or rospy.get_rostime()
        self.left_info.header.frame_id = frame_id
        self.left_info.height = self.rgb_height
        self.left_info.width = self.rgb_width
        self.left_info.distortion_model = "rational_polynomial"
        self.left_info.D = D
        self.left_info.K = K
        self.left_info.R = R
        self.left_info.P = P
        self.left_info.binning_x = 0
        self.left_info.binning_y = 0
        self.left_info.roi = roi

    def right_camera_info(self,D,K,R,P,frame_id):
        roi = RegionOfInterest()
        self.right_info = CameraInfo()
        self.right_info.header.stamp = rospy.Time.now() # or rospy.get_rostime()
        self.right_info.header.frame_id = frame_id
        self.right_info.height = self.rgb_height
        self.right_info.width = self.rgb_width
        self.right_info.distortion_model = "rational_polynomial"
        self.right_info.D = D
        self.right_info.K = K
        self.right_info.R = R
        self.right_info.P = P
        self.right_info.binning_x = 0
        self.right_info.binning_y = 0
        self.right_info.roi = roi
        
    def rgb_camera_info(self,D,K,R,P,frame_id):
        roi = RegionOfInterest()
        self.rgb_info = CameraInfo()
        self.rgb_info.header.stamp = rospy.Time.now() # or rospy.get_rostime()
        self.rgb_info.header.frame_id = frame_id
        self.rgb_info.height = self.rgb_height
        self.rgb_info.width = self.rgb_width
        self.rgb_info.distortion_model = "rational_polynomial"
        self.rgb_info.D = D
        self.rgb_info.K = K
        self.rgb_info.R = R
        self.rgb_info.P = P
        self.rgb_info.binning_x = 0
        self.rgb_info.binning_y = 0
        self.rgb_info.roi = roi

    def depth_camera_info(self,D,K,R,P,frame_id):
        roi = RegionOfInterest()
        self.depth_info = CameraInfo()
        self.depth_info.header.stamp = rospy.Time.now() # or rospy.get_rostime()
        self.depth_info.header.frame_id = frame_id
        self.depth_info.height = self.rgb_height
        self.depth_info.width = self.rgb_width
        self.depth_info.distortion_model = "rational_polynomial"
        self.depth_info.D = D
        self.depth_info.K = K
        self.depth_info.R = R
        self.depth_info.P = P
        self.depth_info.binning_x = 0
        self.depth_info.binning_y = 0
        self.depth_info.roi = roi

    def publish_tf(self):
        br = tf.TransformBroadcaster()
        br.sendTransform((0.0, 0.0, 0.0),
                        (-0.5, 0.5, -0.5, 0.5),
                        rospy.Time.now(),
                        "oakd_depth_camera_optical_frame",
                        "oakd_frame")
        br.sendTransform((0.0, 0.0, 0.0),
                        (-0.5, 0.5, -0.5, 0.5),
                        rospy.Time.now(),
                        "oakd_rgb_camera_optical_frame",
                        "oakd_frame")
        br.sendTransform((0.0, -0.0375, 0.0),
                        (-0.5, 0.5, -0.5, 0.5),
                        rospy.Time.now(),
                        "oakd_right_camera_optical_frame",
                        "oakd_frame")
        br.sendTransform((0.0, 0.0375, 0.0),
                        (-0.5, 0.5, -0.5, 0.5),
                        rospy.Time.now(),
                        "oakd_left_camera_optical_frame",
                        "oakd_frame")
        # br.sendTransform((0.0, 0.0, 0.0),
        #                 (0.0, 0.0, 0.0, 1.0),
        #                 rospy.Time.now(),
        #                 "oakd_frame",
        #                 "odom")
        # br.sendTransform((0.0, 0.0, 0.0),
        #                 (0.0, 0.0, 0.0, 1.0),
        #                 rospy.Time.now(),
        #                 "odom",
        #                 "map")

    def main(self):
        # Connect to device and start self.pipeline
        with dai.Device(self.pipeline, usb2Mode = True) as device:
            # calibFile = str((Path(__file__).parent / Path(f"calib_{device.getMxId()}.json")).resolve().absolute())
            # if len(sys.argv) > 1:
            #     calibFile = sys.argv[1]

            # calibData = device.readCalibration()
            # calibData.eepromToJsonFile(calibFile)

            # M_left = np.array(calibData.getCameraIntrinsics(calibData.getStereoLeftCameraId(), self.depth_width, self.depth_height))
            # M_right = np.array(calibData.getCameraIntrinsics(calibData.getStereoRightCameraId(), self.depth_width, self.depth_height))

            # # left camera info, distortion model, projection matrix, intrinsic camera matrix
            # D_left = np.array(calibData.getDistortionCoefficients(dai.CameraBoardSocket.LEFT))
            # K_left = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.LEFT, self.depth_width, self.depth_height))
            # R1 = np.array(calibData.getStereoLeftRectificationRotation())
            # left_D = [data for data in D_left]
            # left_D = left_D[:8]
            # left_K = [K_left[0][0],K_left[0][1],K_left[0][2],K_left[1][0],K_left[1][1],K_left[1][2],K_left[2][0],K_left[2][1],K_left[2][2]]
            # left_P = [K_left[0][0],K_left[0][1],K_left[0][2],33.634437581578965,K_left[1][0],K_left[1][1],K_left[1][2],0.0,K_left[2][0],K_left[2][1],K_left[2][2],0.0]

            # # right camera info, distortion model, projection matrix, intrinsic camera matrix
            # D_right = np.array(calibData.getDistortionCoefficients(dai.CameraBoardSocket.RIGHT))
            # K_right = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.RIGHT, self.depth_width, self.depth_height))
            # R2 = np.array(calibData.getStereoRightRectificationRotation())
            # right_D = [data for data in D_right]
            # right_D = right_D[:8]
            # right_K = [K_right[0][0],K_right[0][1],K_right[0][2],K_right[1][0],K_right[1][1],K_right[1][2],K_right[2][0],K_right[2][1],K_right[2][2]]
            # right_P = [K_right[0][0],K_right[0][1],K_right[0][2],0.0,K_right[1][0],K_right[1][1],K_right[1][2],0.0,K_right[2][0],K_right[2][1],K_right[2][2],0.0]

            # # rgb camera info, distortion model, projection matrix, intrinsic camera matrix
            # D_rgb = np.array(calibData.getDistortionCoefficients(dai.CameraBoardSocket.RGB))
            # K_rgb = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.RGB, self.rgb_width, self.rgb_height))
            # rgb_D = [data for data in D_rgb]
            # rgb_D = rgb_D[:8]
            # rgb_K = [K_rgb[0][0],K_rgb[0][1],K_rgb[0][2],K_rgb[1][0],K_rgb[1][1],K_rgb[1][2],K_rgb[2][0],K_rgb[2][1],K_rgb[2][2]]
            # rgb_P = [K_rgb[0][0],K_rgb[0][1],K_rgb[0][2],0.0,K_rgb[1][0],K_rgb[1][1],K_rgb[1][2],0.0,K_rgb[2][0],K_rgb[2][1],K_rgb[2][2],0.0]


            # # get left right rgb rectification matrix
            # H_left = np.matmul(np.matmul(M_right, R1), np.linalg.inv(M_left))
            # left_R = [H_left[0][0],H_left[0][1],H_left[0][2],H_left[1][0],H_left[1][1],H_left[1][2],H_left[2][0],H_left[2][1],H_left[2][2]]
            # H_right = np.matmul(np.matmul(M_right, R1), np.linalg.inv(M_right))
            # right_R = [H_right[0][0],H_right[0][1],H_right[0][2],H_right[1][0],H_right[1][1],H_right[1][2],H_right[2][0],H_right[2][1],H_right[2][2]]
            # rgb_R= [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

            # print("H_left: {}".format(H_left))
            # print("H_right: {}".format(H_right))
            # print("left_D: {}".format(left_D))
            # print("left_K: {}".format(left_K))
            # print("left_R: {}".format(left_R))
            # print("left_P: {}".format(left_P))
            # print("right_D: {}".format(right_D))
            # print("right_K: {}".format(right_K))
            # print("right_R: {}".format(right_R))
            # print("right_P: {}".format(right_P))
            # print("len left_D: {}".format(len(left_D)))
            # print("len left_K: {}".format(len(left_K)))
            # print("len left_R: {}".format(len(left_R)))
            # print("len left_P: {}".format(len(left_P)))
            # print("len right_D: {}".format(len(right_D)))
            # print("len right_K: {}".format(len(right_K)))
            # print("len right_R: {}".format(len(right_R)))
            # print("len right_P: {}".format(len(right_P)))

            left_D= [-7.678529739379883, 14.121261596679688, 0.0006415338721126318, 0.00041900016367435455, 29.78580093383789, -7.689117908477783, 14.212541580200195, 29.494033813476562]
            left_K= [454.1698913574219, 0.0, 320.5162658691406, 0.0, 454.1698913574219, 220.9326629638672, 0.0, 0.0, 1.0]
            left_R= [0.9988057613372803, -0.005133225582540035, -0.048586953431367874, 0.004993360489606857, 0.999983012676239, -0.002999595133587718, 0.04860152676701546, 0.0027534007094800472, 0.9988144636154175]
            left_P= [450.0043640136719, 0.0, 300.8768005371094, 33.634437581578965, 0.0, 450.0043640136719, 227.22714233398438, 0.0, 0.0, 0.0, 1.0, 0.0]
            right_D= [-7.938346862792969, 35.663291931152344, 0.0014144983142614365, 0.0003827648179139942, -27.760452270507812, -7.938197135925293, 35.60935974121094, -27.798263549804688]
            right_K= [450.0043640136719, 0.0, 300.8768005371094, 0.0, 450.0043640136719, 227.22714233398438, 0.0, 0.0, 1.0]
            right_R= [0.999995231628418, -0.0010762695455923676, 0.002901612315326929, 0.0010679160477593541, 0.9999952912330627, 0.002878909930586815, -0.002904697088524699, -0.00287579745054245, 0.9999916553497314]
            right_P= [450.0043640136719, 0.0, 300.8768005371094, 0.0, 0.0, 450.0043640136719, 227.22714233398438, 0.0, 0.0, 0.0, 1.0, 0.0]
            rgb_D= [5.5736403465271, 4.625349521636963, -0.0008459454984404147, 0.0009532927651889622, 42.81727981567383, 5.4898786544799805, 3.844762086868286, 44.160743713378906]
            rgb_K= [501.28552246, 0.0, 321.96432495, 0.0, 501.28552246, 177.19174194, 0.0, 0.0, 1.0]
            rgb_R= [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            rgb_P= [501.28552246, 0.0, 321.96432495, 0.0, 0.0, 501.28552246, 177.19174194, 0.0, 0.0, 0.0, 1.0, 0.0]
            depth_D= [5.5736403465271, 4.625349521636963, -0.0008459454984404147, 0.0009532927651889622, 42.81727981567383, 5.4898786544799805, 3.844762086868286, 44.160743713378906]
            depth_K= [501.28552246, 0.0, 321.96432495, 0.0, 501.28552246, 177.19174194, 0.0, 0.0, 1.0]
            depth_R= [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            depth_P= [501.28552246, 0.0, 321.96432495, 0.0, 0.0, 501.28552246, 177.19174194, 0.0, 0.0, 0.0, 1.0, 0.0]

            # Output queues will be used to get the rgb frames and nn data from the outputs defined above
            previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
            rectified_left_camera_Queue = device.getOutputQueue(name="rectified_left", maxSize=4, blocking=False)
            rectified_right_camera_Queue = device.getOutputQueue(name="rectified_right", maxSize=4, blocking=False)

            while not rospy.is_shutdown():
                self.publish_tf()
                inPreview = previewQueue.get()
                depth = depthQueue.get()
                rectified_left = rectified_left_camera_Queue.get()
                rectified_right = rectified_right_camera_Queue.get()

                frame = inPreview.getCvFrame()
                rectified_left_image = rectified_left.getCvFrame()
                rectified_right_image = rectified_right.getCvFrame()
                depthFrame = depth.getFrame() # depthFrame values are in millimeters

                depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
                depthFrameColor = cv2.equalizeHist(depthFrameColor)
                depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT) # COLORMAP_HOT or COLORMAP_JET

                self.rgb_pub.publish(self.bridge.cv2_to_imgmsg(frame,frame_id="oakd_rgb_camera_optical_frame",encoding="bgr8"))
                self.depth_pub.publish(self.bridge.cv2_to_imgmsg(depthFrame,frame_id="oakd_depth_camera_optical_frame",encoding="mono16")) # depthFrameColor,depthFrame
                self.left_pub.publish(self.bridge.cv2_to_imgmsg(rectified_left_image,frame_id="oakd_left_camera_optical_frame",encoding="mono8"))
                self.right_pub.publish(self.bridge.cv2_to_imgmsg(rectified_right_image,frame_id="oakd_right_camera_optical_frame",encoding="mono8"))

                self.left_camera_info(left_D,left_K,left_R,left_P,frame_id="oakd_left_camera_optical_frame")
                self.right_camera_info(right_D,right_K,right_R,right_P,frame_id="oakd_right_camera_optical_frame")
                self.rgb_camera_info(rgb_D,rgb_K,rgb_R,rgb_P,frame_id="oakd_rgb_camera_optical_frame")
                self.depth_camera_info(depth_D,depth_K,depth_R,depth_P,frame_id="oakd_depth_camera_optical_frame")
                self.left_info_pub.publish(self.left_info)
                self.right_info_pub.publish(self.right_info)
                self.rgb_info_pub.publish(self.rgb_info)
                self.depth_info_pub.publish(self.depth_info)
                # cv2.imshow("left", left_image)
                # cv2.imshow("right", right_image)
                # cv2.imshow("depth", depthFrameColor)
                # cv2.imshow("rgb", frame)
                # if cv2.waitKey(1) == ord('q'):
                #     break
                self.loop_rate.sleep()

if __name__ == '__main__':
    depth_camera = oakd_lite()
    depth_camera.main()

# for rtabmap ros launch 
# case 1: mono camera
# roslaunch rtabmap_ros rtabmap.launch args:="--delete_db_on_start" depth_topic:=/oakd_lite/depth/image rgb_topic:=/oakd_lite/rgb/image camera_info_topic:=/oakd_lite/rgb/camera_info frame_id:=oakd_frame approx_sync:=true rviz:=true
# case 2: stereo camera
# roslaunch rtabmap_ros rtabmap.launch args:="--delete_db_on_start" stereo:=true left_image_topic:=/oakd_lite/left/image_rect right_image_topic:=/oakd_lite/right/image_rect left_camera_info_topic:=/oakd_lite/left/camera_info right_camera_info_topic:=/oakd_lite/right/camera_info frame_id:=oakd_frame approx_sync:=true rviz:=true
# case 3: point cloud
# rosrun nodelet nodelet standalone rtabmap_ros/point_cloud_xyz _approx_sync:=true /depth/image:=/oakd_lite/depth/image /depth/camera_info:=/oakd_lite/depth/camera_info _decimation:=4
# roslaunch rtabmap_ros rtabmap.launch rtabmap_args:="--delete_db_on_start --Icp/VoxelSize 0.05 --Icp/PointToPlaneRadius 0 --Icp/PointToPlaneK 20 --Icp/CorrespondenceRatio 0.2 --Icp/PMOutlierRatio 0.65 --Icp/Epsilon 0.005 --Icp/PointToPlaneMinComplexity 0 --Odom/ScanKeyFrameThr 0.7 --OdomF2M/ScanMaxSize 15000 --Optimizer/GravitySigma 0.3 --RGBD/ProximityPathMaxNeighbors 1 --Reg/Strategy 1" icp_odometry:=true scan_cloud_topic:=/cloud subscribe_scan_cloud:=true depth_topic:=/oakd_lite/depth/image rgb_topic:=/oakd_lite/rgb/image camera_info_topic:=/oakd_lite/rgb/camera_info approx_sync:=true rviz:=true frame_id:=oakd_frame

# for oakd_development ros launch
# case 1: mono camera
# roslaunch oakd_development oakd_lite_mapping_rgbd_odometry.launch
# case 2: stereo camera
# roslaunch oakd_development oakd_lite_mapping_stereo_odometry.launch
# case 3: point cloud
# roslaunch oakd_development oakd_lite_mapping_icp_odometry.launch