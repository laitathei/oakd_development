import sys
import torch
import blobconverter
import shutil

SavePath = "./output/model_20.pth"
model = torch.load(SavePath)
device = torch.device('cpu')
model.to(device)
model.eval() # put the model in evaluation mode

# Convert to OpenVINO IR
sys.path.append('./openvino_contrib/modules/mo_pytorch')

import mo_pytorch
mo_pytorch.convert(model, input_shape=[1, 3, 360, 640], model_name='maskrcnn_resnet50_360_640', scale = 255, reverse_input_channels=True)

xml_file = "./maskrcnn_resnet50_360_640.xml"
bin_file = "./maskrcnn_resnet50_360_640.bin"
blob_path = blobconverter.from_openvino(
    xml=xml_file,
    bin=bin_file,
    data_type="FP16",
    shaves=6,
    version="2021.4"
)

shutil.copy(str(blob_path), "./output")
print("Done export openvino")