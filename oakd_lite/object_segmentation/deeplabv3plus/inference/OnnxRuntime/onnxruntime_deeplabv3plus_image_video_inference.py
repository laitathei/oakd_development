import argparse
import os
import numpy as np
import onnxruntime
import cv2
import time

def decode_seg_map_sequence(label_masks,num_class, dataset='grass'): # change
    rgb_masks = []
    for label_mask in label_masks:
        rgb_mask = decode_segmap(label_mask, dataset, num_class)
        rgb_masks.append(rgb_mask)
    rgb_masks = np.array(rgb_masks).transpose([0, 3, 1, 2])
    return rgb_masks

def decode_segmap(label_mask, dataset, num_class,plot=False):
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
        n_classes = num_class
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
    return np.asarray([[0,0,0],[128,0,0],[0,128,0],[0,0,128],[256,0,0],[0,256,0],[0,0,256]]) # change with your desired mask color for your custom dataset if you want

def composed_transforms(sample,mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
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

parser = argparse.ArgumentParser(description="OnnxRuntim DeeplabV3Plus Inferencing")
parser.add_argument('--in-path', type=str, required=True, help='image to test')
parser.add_argument('--weight', type=str, default='./models/model_352_640.onnx',
                    help='saved model')
parser.add_argument('--num_class', type=int, help='number of class')

args = parser.parse_args()

sess = onnxruntime.InferenceSession(args.weight) # model_height_width.onnx
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
dataset = "custom_dataset"

for name in os.listdir(args.in_path):
    if name.endswith('.jpg'):
        s_time = time.time()
        origin_image = cv2.imread(args.in_path+"/"+name)
        origin_image = cv2.cvtColor(origin_image,cv2.COLOR_BGR2RGB)
        mask_image = cv2.imread(args.in_path+"/"+name)
        mask_image = cv2.cvtColor(mask_image,cv2.COLOR_BGR2GRAY)
        sample = {'image': origin_image, 'label': mask_image}
        tensor_in = composed_transforms(sample)['image']
        tensor_in = np.expand_dims(tensor_in, axis=0) # (1, 3, 352, 640)
        pred_onnx = sess.run([label_name],{input_name: tensor_in.astype(np.float32)})[0] # (1, 2, 352, 640)
                    # np.argmax(pred_onnx[:3],axis=1) return the max value indice in each row
        decode = decode_seg_map_sequence(np.argmax(pred_onnx[:3],axis=1),args.num_class,dataset=dataset) # (1, 3, 352, 640)
        #print(decode)
        grid_image = np.squeeze(decode, axis=0)
        grid_image = grid_image*255 # mul(255)
        grid_image = grid_image+0.5 # add_(0.5)
        grid_image = np.clip(grid_image,0,255) # clamp_(0, 255)
        grid_image_ndarr = np.array(grid_image,dtype=np.uint8)
        grid_image_ndarr = np.transpose(grid_image_ndarr,(1,2,0))
        origin_image = np.array(origin_image,dtype=np.uint8)
        mix = cv2.addWeighted(origin_image,0.8,grid_image_ndarr,1.0,0)
        cv2.imshow("mask",cv2.cvtColor(grid_image_ndarr,cv2.COLOR_BGR2RGB))
        cv2.imshow("original",cv2.cvtColor(origin_image,cv2.COLOR_BGR2RGB))
        cv2.imshow("mix",cv2.cvtColor(mix,cv2.COLOR_BGR2RGB))
        cv2.waitKey(0)
        u_time = time.time()
        img_time = u_time-s_time
        print("image:{} time: {} ".format(name,img_time))
    elif name.endswith('.mp4'):
        cap = cv2.VideoCapture(args.in_path+"/"+name)
        while cap.isOpened():
            s_time = time.time()
            ret, frame = cap.read()
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            origin_image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            mask_image = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            origin_image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            target = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            sample = {'image': origin_image, 'label': mask_image}
            tensor_in = composed_transforms(sample)['image']
            tensor_in = np.expand_dims(tensor_in, axis=0) # (1, 3, 352, 640)
            pred_onnx = sess.run([label_name],{input_name: tensor_in.astype(np.float32)})[0] # (1, 2, 352, 640)
                        # np.argmax(pred_onnx[:3],axis=1) return the max value indice in each row
            decode = decode_seg_map_sequence(np.argmax(pred_onnx[:3],axis=1),args.num_class,dataset=dataset) # (1, 3, 352, 640)
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
            origin_image = cv2.putText(origin_image, "fps: %.2f"%(fps), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            grid_image_ndarr = cv2.putText(grid_image_ndarr, "fps: %.2f"%(fps), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            mix = cv2.putText(mix, "fps: %.2f"%(fps), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("mask",cv2.cvtColor(grid_image_ndarr,cv2.COLOR_BGR2RGB))
            cv2.imshow("original",cv2.cvtColor(origin_image,cv2.COLOR_BGR2RGB))
            cv2.imshow("mix",cv2.cvtColor(mix,cv2.COLOR_BGR2RGB))
            cv2.waitKey(1)

# python3 onnxruntime_deeplabv3plus_image_video_inference.py --in-path ./source --weight ./models/model_352_640.onnx --num_class 6
