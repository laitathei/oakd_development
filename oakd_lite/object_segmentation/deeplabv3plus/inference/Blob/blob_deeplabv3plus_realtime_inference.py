#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("-nn", "--nn_model", help="select model path for inference", type=str)
parser.add_argument("--h", help="inference image height", type=int)
parser.add_argument("--w", help="inference image width", type=int)
parser.add_argument("--num_class", help="number of class", type=int)
args = parser.parse_args()

num_of_classes = args.num_class
nn_path = args.nn_model 
image_height = args.h
image_width = args.w

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

def decode_seg_map_sequence(label_masks, n_classes, dataset='custom_dataset'): # change
    rgb_masks = []
    #print(label_masks.shape) # need (1, 360, 640)
    for label_mask in label_masks:
        rgb_mask = decode_segmap(label_mask, dataset, n_classes)
        rgb_masks.append(rgb_mask)
    rgb_masks = np.array(rgb_masks).transpose([0, 3, 1, 2])
    return rgb_masks

def decode_segmap(label_mask, dataset, n_classes, plot=False):
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
        label_colours = get_custom_dataset_labels()
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

def get_custom_dataset_labels():
    return np.asarray([[0,0,0],[128,0,0]]) # change with your desired mask color for your custom dataset if you want


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
        origin_image = frame
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
        lay1 = np.array(in_nn.getFirstLayerFp16()).reshape(1, num_of_classes, image_height,image_width) # in_nn.getFirstLayerFp16() = 1 x num_of_classes x h x w
        lay1 = np.argmax(lay1[:3],axis=1) # get masked class indices (1, h, w)
        found_classes = np.unique(np.squeeze(lay1, axis=0)) # 1d array
        output_colors = decode_seg_map_sequence(lay1,num_of_classes,dataset="custom_dataset")
        output_colors = np.squeeze(output_colors, axis=0)
        output_colors = output_colors*255 # mul(255)
        output_colors = output_colors+0.5 # add_(0.5)
        output_colors = np.clip(output_colors,0,255) # clamp_(0, 255)
        output_colors = np.array(output_colors,dtype=np.uint8)
        output_colors = np.transpose(output_colors,(1,2,0)) # (h, w, 3)
        output_colors = cv2.cvtColor(output_colors,cv2.COLOR_BGR2RGB)
        #output_colors = np.array(output_colors).transpose([1, 0, 2]) # (h, w, 3)
        frame = cv2.addWeighted(frame,1, output_colors,0.4,0)
        cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (0, 255, 0))
        cv2.putText(frame, "Found classes {}".format(found_classes), (2, 10), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (0, 255, 0))
        cv2.imshow("nn_input", frame)
        cv2.imshow("mask image",output_colors)
        cv2.imshow("origin image",origin_image)
        #t2 = time.time()
        #print("Duration : {}".format(t2-t1))
        counter+=1
        if (time.time() - start_time) > 1 :
            fps = counter / (time.time() - start_time)

            counter = 0
            start_time = time.time()


        if cv2.waitKey(1) == ord('q'):
            break

# python3 blob_deeplabv3plus_realtime_inference.py --nn_model ./models/model_352_640.blob --h 640 --w 352 --num_class 6
