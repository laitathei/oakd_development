import argparse
import json
import os
from os import path
from symbol import file_input
import PIL.Image
from labelme import utils
import base64
import glob
import shutil
import pathlib
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dataset_dir', default=None)
    parser.add_argument('--output_dataset_dir', default=None)
    parser.add_argument('--image_name', default=None)
    parser.add_argument('--train_val_ratio', default=None)
    args = parser.parse_args()
 
    input_dataset_dir = args.input_dataset_dir
    input_dataset_dir = input_dataset_dir+"/"
    ouput_dataset_dir = args.output_dataset_dir
    ouput_dataset_dir = ouput_dataset_dir+"/"
    image_name = args.image_name
    train_val_ratio = float(args.train_val_ratio)


    # prepare train val data folder path
    train_folder = os.path.join(ouput_dataset_dir, 'train/')
    val_folder = os.path.join(ouput_dataset_dir, 'val/')
    train_visualize_image_folder = os.path.join(train_folder, 'visualize_image/')
    train_origin_image_folder = os.path.join(train_folder, 'origin_image/')
    train_mask_image_folder = os.path.join(train_folder, 'mask_image/')
    train_json_folder = os.path.join(train_folder, 'json/')
    val_visualize_image_folder = os.path.join(val_folder, 'visualize_image/')
    val_origin_image_folder = os.path.join(val_folder, 'origin_image/')
    val_mask_image_folder = os.path.join(val_folder, 'mask_image/')
    val_json_folder = os.path.join(val_folder, 'json/')

    # create those data folder if not exist
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    if not os.path.exists(val_folder):
        os.makedirs(val_folder)
    if not os.path.exists(train_visualize_image_folder):
        os.makedirs(train_visualize_image_folder)
    if not os.path.exists(train_origin_image_folder):
        os.makedirs(train_origin_image_folder)
    if not os.path.exists(train_mask_image_folder):
        os.makedirs(train_mask_image_folder)
    if not os.path.exists(train_json_folder):
        os.makedirs(train_json_folder)
    if not os.path.exists(val_visualize_image_folder):
        os.makedirs(val_visualize_image_folder)
    if not os.path.exists(val_origin_image_folder):
        os.makedirs(val_origin_image_folder)
    if not os.path.exists(val_mask_image_folder):
        os.makedirs(val_mask_image_folder)
    if not os.path.exists(val_json_folder):
        os.makedirs(val_json_folder)

    count = os.listdir(input_dataset_dir)
    jpg_file_list = glob.glob(os.path.join(input_dataset_dir, "*.jpg"))
    json_file_list = glob.glob(os.path.join(input_dataset_dir, "*.json"))

    # make copy of json,visualize_image,mask_image,origin_image to output dataset directory
    if train_val_ratio != 0:
        train_idxs, val_idxs = train_test_split(range(len(jpg_file_list)), test_size=train_val_ratio)
        for i in range (len(train_idxs)):
            json_file = input_dataset_dir+image_name+str(train_idxs[i])+".json"
            if os.path.isfile(json_file):
                data = json.load(open(json_file))
                if data['imageData']:
                    imageData = data['imageData']
                else:
                    imagePath = os.path.join(os.path.dirname(json_file), data['imagePath'])
                    with open(imagePath, 'rb') as f:
                        imageData = f.read()
                        imageData = base64.b64encode(imageData).decode('utf-8')
                img = utils.img_b64_to_arr(imageData)
                label_name_to_value = {'_background_': 0}
                for shape in data['shapes']:
                    label_name = shape['label']
                    if label_name in label_name_to_value:
                        label_value = label_name_to_value[label_name]
                    else:
                        label_value = len(label_name_to_value)
                        label_name_to_value[label_name] = label_value
                
                lbl = utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)
                
                captions = ['{}: {}'.format(lv, ln) for ln, lv in label_name_to_value.items()]
                lbl_viz = utils.draw_label(lbl, img, captions)

                utils.lblsave(train_mask_image_folder+image_name+str(train_idxs[i])+".png", lbl) # move train mask image
                PIL.Image.fromarray(lbl_viz).save(train_visualize_image_folder+image_name+str(train_idxs[i])+".png") # move train visual image
    
            if path.exists(input_dataset_dir+"/"+image_name+str(train_idxs[i])+".jpg"):
                shutil.move(input_dataset_dir+"/"+image_name+str(train_idxs[i])+".jpg", train_origin_image_folder) # move train origin image
            if path.exists(input_dataset_dir+"/"+image_name+str(train_idxs[i])+".json"):
                shutil.move(input_dataset_dir+"/"+image_name+str(train_idxs[i])+".json", train_json_folder) # move train json
            print("Done %s processing" %json_file)

        for i in range (len(val_idxs)):
            json_file = input_dataset_dir+image_name+str(val_idxs[i])+".json"
            if os.path.isfile(json_file):
                data = json.load(open(json_file))
                if data['imageData']:
                    imageData = data['imageData']
                else:
                    imagePath = os.path.join(os.path.dirname(json_file), data['imagePath'])
                    with open(imagePath, 'rb') as f:
                        imageData = f.read()
                        imageData = base64.b64encode(imageData).decode('utf-8')
                img = utils.img_b64_to_arr(imageData)
                label_name_to_value = {'_background_': 0}
                for shape in data['shapes']:
                    label_name = shape['label']
                    if label_name in label_name_to_value:
                        label_value = label_name_to_value[label_name]
                    else:
                        label_value = len(label_name_to_value)
                        label_name_to_value[label_name] = label_value
                
                lbl = utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)
                
                captions = ['{}: {}'.format(lv, ln) for ln, lv in label_name_to_value.items()]
                lbl_viz = utils.draw_label(lbl, img, captions)

                utils.lblsave(val_mask_image_folder+image_name+str(val_idxs[i])+".png", lbl) # move val mask image
                PIL.Image.fromarray(lbl_viz).save(val_visualize_image_folder+image_name+str(val_idxs[i])+".png") # move val visual image
    
            if path.exists(input_dataset_dir+"/"+image_name+str(val_idxs[i])+".jpg"):
                shutil.move(input_dataset_dir+"/"+image_name+str(val_idxs[i])+".jpg", val_origin_image_folder) # move val origin image
            if path.exists(input_dataset_dir+"/"+image_name+str(val_idxs[i])+".json"):
                shutil.move(input_dataset_dir+"/"+image_name+str(val_idxs[i])+".json", val_json_folder) # move val json
            print("Done %s processing" %json_file)

    else:
        train_idxs = []
        for i in range (len(jpg_file_list)):
            train_idxs.append(i)
        for i in range (len(train_idxs)):
            json_file = input_dataset_dir+image_name+str(train_idxs[i])+".json"
            if os.path.isfile(json_file):
                data = json.load(open(json_file))
                if data['imageData']:
                    imageData = data['imageData']
                else:
                    imagePath = os.path.join(os.path.dirname(json_file), data['imagePath'])
                    with open(imagePath, 'rb') as f:
                        imageData = f.read()
                        imageData = base64.b64encode(imageData).decode('utf-8')
                img = utils.img_b64_to_arr(imageData)
                label_name_to_value = {'_background_': 0}
                for shape in data['shapes']:
                    label_name = shape['label']
                    if label_name in label_name_to_value:
                        label_value = label_name_to_value[label_name]
                    else:
                        label_value = len(label_name_to_value)
                        label_name_to_value[label_name] = label_value
                
                lbl = utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)
                
                captions = ['{}: {}'.format(lv, ln) for ln, lv in label_name_to_value.items()]
                lbl_viz = utils.draw_label(lbl, img, captions)

                utils.lblsave(train_mask_image_folder+image_name+str(train_idxs[i])+".png", lbl) # move train mask image
                PIL.Image.fromarray(lbl_viz).save(train_visualize_image_folder+image_name+str(train_idxs[i])+".png") # move train visual image
    
            if path.exists(input_dataset_dir+"/"+image_name+str(train_idxs[i])+".jpg"):
                shutil.move(input_dataset_dir+"/"+image_name+str(train_idxs[i])+".jpg", train_origin_image_folder) # move train origin image
            if path.exists(input_dataset_dir+"/"+image_name+str(train_idxs[i])+".json"):
                shutil.move(input_dataset_dir+"/"+image_name+str(train_idxs[i])+".json", train_json_folder) # move train json
            print("Done %s processing" %json_file)


if __name__ == '__main__':
    main()
