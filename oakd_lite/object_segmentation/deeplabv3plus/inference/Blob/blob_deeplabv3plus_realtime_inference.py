#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
import argparse
import time

num_of_classes = 2 # define the number of classes in the dataset

parser = argparse.ArgumentParser()
parser.add_argument("-nn", "--nn_model", help="select model path for inference", type=str)
parser.add_argument("--h", help="inference image height", type=int)
parser.add_argument("--w", help="inference image width", type=int)
args = parser.parse_args()

nn_path = args.nn_model 
image_height = args.h
image_width = args.w

def decode_deeplabv3p(output_tensor):
    output = output_tensor.reshape(image_width,image_height)

    # scale to [0 ... 2555] and apply colormap
    output = np.array(output) * (255/num_of_classes)
    output = output.astype(np.uint8)
    output_colors = cv2.applyColorMap(output, cv2.COLORMAP_JET)

    # reset the color of 0 class
    output_colors[output == 0] = [0,0,0]

    return output_colors

def show_deeplabv3p(output_colors, frame):
    return cv2.addWeighted(frame,1, output_colors,0.4,0)

# Start defining a pipeline
pipeline = dai.Pipeline()

pipeline.setOpenVINOVersion(version = dai.OpenVINO.VERSION_2021_4)

# Define a neural network that will make predictions based on the source frames
detection_nn = pipeline.create(dai.node.NeuralNetwork)
detection_nn.setBlobPath(nn_path)

detection_nn.setNumPoolFrames(4)
detection_nn.input.setBlocking(False)
detection_nn.setNumInferenceThreads(2)


# Define a source - color camera
cam = pipeline.create(dai.node.ColorCamera)
cam.setPreviewSize(image_width,image_height)
cam.setInterleaved(False)
cam.preview.link(detection_nn.input)
cam.setFps(40)

# Create outputs
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("nn_input")
xout_rgb.input.setBlocking(False)

detection_nn.passthrough.link(xout_rgb.input)

xout_nn = pipeline.create(dai.node.XLinkOut)
xout_nn.setStreamName("nn")
xout_nn.input.setBlocking(False)

detection_nn.out.link(xout_nn.input)

# Pipeline defined, now the device is assigned and pipeline is started
with dai.Device() as device:
    cams = device.getConnectedCameras()
    device.startPipeline(pipeline)

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    q_nn_input = device.getOutputQueue(name="nn_input", maxSize=4, blocking=False)
    q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    start_time = time.time()
    counter = 0
    fps = 0
    layer_info_printed = False
    #t1 = time.time()
    while True:
        # instead of get (blocking) used tryGet (nonblocking) which will return the available data or None otherwise
        in_nn_input = q_nn_input.get()
        in_nn = q_nn.get()

        frame = in_nn_input.getCvFrame()
        layers = in_nn.getAllLayers()
        layers_name = in_nn.getAllLayerNames() # ['output']

        # for layer_nr, layer in enumerate(layers):
        #     print("_______________________________")
        #     print(f"Layer {layer_nr}")
        #     print(f"Name: {layer.name}")
        #     print(f"Order: {layer.order}")
        #     print(f"dataType: {layer.dataType}")
        #     #dims = layer.dims[::-1] # reverse dimensions
        #     print(f"dims: {layer.dims}")

        # get layer1 data
        lay1 = np.array(in_nn.getFirstLayerFp16()).reshape(1, num_of_classes, image_width,image_height) # in_nn.getFirstLayerFp16() = 1 x num_of_classes x h x w
        lay1 = np.argmax(lay1[:3],axis=1) # get masked class indices (1, h, w)
        lay1 = np.squeeze(lay1, axis=0) # (h, w)
        found_classes = np.unique(lay1) # 1d array
        output_colors = decode_deeplabv3p(lay1) # (w, h, 3)
        output_colors = np.array(output_colors).transpose([1, 0, 2]) # (h, w, 3)
        frame = show_deeplabv3p(output_colors, frame)
        cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255, 0, 0))
        cv2.putText(frame, "Found classes {}".format(found_classes), (2, 10), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255, 0, 0))
        cv2.imshow("nn_input", frame)
        #t2 = time.time()
        #print("Duration : {}".format(t2-t1))
        counter+=1
        if (time.time() - start_time) > 1 :
            fps = counter / (time.time() - start_time)

            counter = 0
            start_time = time.time()


        if cv2.waitKey(1) == ord('q'):
            break

# python3 blob_deeplabv3plus_realtime_inference.py --nn_model ./models/model_352_640.blob --h 640 --w 352
