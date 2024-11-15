import torch
from torchinfo import summary
from torchvision.models import AlexNet,efficientnet  # 导入AlexNet官方模型
import netron  # 导入Netron库

model = AlexNet().to('cuda')  # 加载AlexNet模型到GPU
print(model)    # 原始的打印模型结构

summary(model, input_size=(64, 3, 224, 224))  # 使用torchinfo打印模型结构和参数信息
#  torchinfo 的 summary 函数的 input_size 应该是一个元组或列表，表示输入张量的形状，而不是一个实际的张量。

torch.onnx.export(model, torch.ones((1,3,224,224)).to('cuda'), f='AlexNet.onnx')   #导出 .onnx 文件
netron.start('AlexNet.onnx') #使用Netron库展示结构图

model = efficientnet.efficientnet_b0().to('cuda')  # 加载EfficientNet-B0模型到GPU
torch.onnx.export(model, torch.ones((1,3,224,224)).to('cuda'),"efficientnet_b0.onnx", opset_version=11)   #导出 .onnx 文件
netron.start('efficientnet_b0.onnx') #使用Netron库展示结构图
