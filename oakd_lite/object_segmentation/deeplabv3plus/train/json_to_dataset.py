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

    # prepare train val data folder path
    ImageSets_folder = os.path.join(ouput_dataset_dir, 'ImageSets/')
    Segmentation_folder = os.path.join(ImageSets_folder, 'Segmentation/')
    JPEGImages_folder = os.path.join(ouput_dataset_dir, 'JPEGImages/')
    Json_folder = os.path.join(ouput_dataset_dir, 'json/')
    SegmentationClass_folder = os.path.join(ouput_dataset_dir, 'SegmentationClass/')
    visualize_image_folder = os.path.join(ouput_dataset_dir, 'visualize_image/')

    # create those data folder if not exist
    if not os.path.exists(ImageSets_folder):
        os.makedirs(ImageSets_folder)
    if not os.path.exists(Segmentation_folder):
        os.makedirs(Segmentation_folder)
    if not os.path.exists(JPEGImages_folder):
        os.makedirs(JPEGImages_folder)
    if not os.path.exists(Json_folder):
        os.makedirs(Json_folder)
    if not os.path.exists(SegmentationClass_folder):
        os.makedirs(SegmentationClass_folder)
    if not os.path.exists(visualize_image_folder):
        os.makedirs(visualize_image_folder)

    count = os.listdir(input_dataset_dir)
    jpg_file_list = glob.glob(os.path.join(input_dataset_dir, "*.jpg"))
    png_file_list = glob.glob(os.path.join(input_dataset_dir, "*.jpg"))
    json_file_list = glob.glob(os.path.join(input_dataset_dir, "*.json"))
    # make copy of json,visualize_image,mask_image,origin_image to output dataset directory
    train_idxs = []
    for i in range (len(json_file_list)):
        json_file = json_file_list[i].replace(input_dataset_dir,"").replace(".json","")
        if os.path.isfile(json_file_list[i]):
            data = json.load(open(json_file_list[i]))
            if data['imageData']:
                imageData = data['imageData']
            else:
                imagePath = os.path.join(os.path.dirname(json_file), data['imagePath'])
                imagePath = os.path.join(input_dataset_dir, imagePath)
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

            utils.lblsave(SegmentationClass_folder+json_file+".png", lbl) # move SegmentationClass_folder
            PIL.Image.fromarray(lbl_viz).save(visualize_image_folder+json_file+".png") # move visualize_image folder

        if path.exists(input_dataset_dir+"/"+json_file+".jpg"):
            shutil.move(input_dataset_dir+"/"+json_file+".jpg", JPEGImages_folder) # move JPEGImages folder
        if path.exists(input_dataset_dir+"/"+json_file+".png"): # if exist png, convert it to jpg
            im = PIL.Image.open(input_dataset_dir+"/"+json_file+".png")
            im = im.convert("RGB")
            im.save(input_dataset_dir+"/"+json_file+".jpg",quality=95)
            shutil.move(input_dataset_dir+"/"+json_file+".jpg", JPEGImages_folder) # move JPEGImages folder
            os.remove(input_dataset_dir+"/"+json_file+".png") # delete the png file
        if path.exists(input_dataset_dir+"/"+json_file+".json"):
            shutil.move(input_dataset_dir+"/"+json_file+".json", Json_folder) # move Json folder
        print("Done %s processing" %json_file)


if __name__ == '__main__':
    main()
