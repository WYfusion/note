import netron
import torch
from torchvision.models import shufflenet_v2_x1_0
from torchinfo import summary

# Define the model  
model = shufflenet_v2_x1_0(pretrained=True)

# Check the model summary
summary(model, input_size=(1, 3, 224, 224))
torch.onnx.export(model, torch.ones((1,3,224,224)).to('cuda'), f='shufflenet_v2_x1_0.onnx')   #导出 .onnx 文件
netron.start('shufflenet_v2_x1_0.onnx') #使用Netron库展示结构图