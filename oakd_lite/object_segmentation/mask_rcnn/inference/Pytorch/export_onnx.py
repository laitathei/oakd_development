import onnx
import os 
import numpy as np
import cv2
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def get_instance_segmentation_model(num_classes):
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

def export():
    class_names = ["__background__", "grass"]
    num_classes = len(class_names)
    model = get_instance_segmentation_model(num_classes)
    pthfile = "./output/model_state_dict_epoch_9.pth"
    input_width = 300 # 640
    input_height = 300 # 352
    loaded_model = torch.load(pthfile)
    model.load_state_dict(loaded_model)
    model.eval()
    dummy_input = torch.randn(1,3,input_height,input_width)
    input_names = ["input"]
    output_names = ["output"]
    torch.onnx.export(model,dummy_input,"model_"+str(input_height)+"_"+str(input_width)+".onnx",verbose=True,input_names=input_names,output_names=output_names,opset_version=11)
    for name in os.listdir(os.getcwd()):
        if name.endswith('.onnx'):
            # Checks
            model_onnx = onnx.load(name)  # load onnx model
            onnx.checker.check_model(model_onnx)  # check onnx model

            # # Metadata
            # d = {'stride': int(max(model.stride)), 'names': model.names}
            # for k, v in d.items():
            #     meta = model_onnx.metadata_props.add()
            #     meta.key, meta.value = k, str(v)
            # onnx.save(model_onnx, name)

            # Simplify
            #cuda = torch.cuda.is_available()
            import onnxsim

            model_onnx, check = onnxsim.simplify(model_onnx)
            assert check, 'assert check failed'
            onnx.save(model_onnx, name)

    print("finish")

if __name__ == "__main__":
    export()