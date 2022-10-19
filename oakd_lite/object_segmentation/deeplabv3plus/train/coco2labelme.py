#!/usr/bin/python3
import argparse
import copy
import json
import os
import os.path as osp

import cv2
import numpy as np
import tqdm

DEFAULT_LABELME = {
    "flags": {},
    "fillColor": [255, 0, 0, 128],
    "lineColor": [0, 255, 0, 128],
    "version": "3.16.7",
    'imageData': None,
}

DEFAULT_SHAPE = {"flags": {}, "line_color": None, "fill_color": None}

def find_all_img_anns(coco):
    img_id_list = []
    anns_list = []
    for img_info in coco['images']:
        img_id_list.append(img_info['id'])
        anns_list.append([])
    for ann in coco['annotations']:
        index = img_id_list.index(ann['image_id'])
        anns_list[index].append(ann)
    return coco['images'], anns_list

def coco2labelme(coco_path, outputs):
    os.makedirs(outputs, exist_ok=True)
    with open(coco_path, 'r') as f:
        coco = json.loads(f.read())
    img_info_list, anns_list = find_all_img_anns(coco)
    for img_info, anns in zip(img_info_list, anns_list):
        labelme_json = copy.deepcopy(DEFAULT_LABELME)
        labelme_json['imageHeight'] = img_info['height']
        labelme_json['imageWidth'] = img_info['width']
        shapes = []
        for ann in anns:
            shape = copy.deepcopy(DEFAULT_SHAPE)
            points = ann['segmentation'][0]
            points = list(zip(points[::2], points[1::2]))
            label = coco['categories'][ann['category_id']]['name']
            if len(points) == 1:
                shape_type = 'point'
            elif len(points) == 2:
                shape_type = 'line'
            elif len(points) <= 0:
                continue
            else:
                shape_type = 'polygon'
            shape['points'] = points
            shape['shape_type'] = shape_type
            shape['label'] = label
            shapes.append(shape)
        labelme_json['shapes'] = shapes
        labelme_json['imagePath'] = osp.relpath(
            osp.join(osp.dirname(coco_path), img_info['file_name']), outputs)
        with open(osp.join(outputs, osp.splitext(osp.basename(img_info['file_name']))[0] + '.json'), 'w') as f:
            f.write(json.dumps(labelme_json, indent=4, sort_keys=True))
    os.remove(coco_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('coco', type=str)
    parser.add_argument('outputs', type=str)
    opt = parser.parse_args()
    coco2labelme(opt.coco, opt.outputs)
