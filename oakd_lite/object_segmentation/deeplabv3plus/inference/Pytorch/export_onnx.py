import onnx
import onnxruntime
import torch
import torchvision.models as models
import torchvision
from modeling.deeplab import *
import argparse
import os 
import numpy as np
import cv2

def export():
    parser = argparse.ArgumentParser(description="Export onnx model")
    parser.add_argument('--ckpt', type=str, default='deeplab-resnet.pth',
                        help='saved model')
    parser.add_argument('--num_class', type=int, help='number of class')
    parser.add_argument('--input_width', type=int, help='image width')
    parser.add_argument('--input_height', type=int, help='image height')
    args = parser.parse_args()

    num_classes = args.num_class
    crop_size = 513
    ckpt=args.ckpt
    freeze_bn=False
    sync_bn=False
    out_stride=16
    backbone = "mobilenet"
    input_width = args.input_width # 640
    input_height = args.input_height # 352

    model = DeepLab(num_classes=num_classes,
                    backbone=backbone,
                    output_stride=out_stride,
                    sync_bn=sync_bn,
                    freeze_bn=freeze_bn)

    ckpt = torch.load(ckpt, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    # dummy_input = cv2.imread("./source/dataset667.jpg")
    # dummy_input = np.array(dummy_input,dtype=np.float32)
    # dummy_input = np.expand_dims(dummy_input, axis=0)
    # dummy_input = np.transpose(dummy_input,(0, 3, 1, 2))
    # dummy_input = torch.from_numpy(dummy_input) # torch.Size([1, 3, 352, 640])
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
