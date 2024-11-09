# Tensorboard

代码位于平地起高楼tensorboard文件夹中

#### 建立方式

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs/runs/test/1") 			# 在当前运行的地址处创建一个这样的路径目录
for x in range(10):
    writer.add_scalar("y=x", x, x)					# ""是图像的标签、X^2是纵轴、x是横轴
    writer.add_scalar("y=x^2", x**2, x)
    writer.add_scalar("y=x^3", x**3, x)

writer.close()
```

- 不同的标签会有不同的图像展现出来。

- 使用相同的标签会在同一张图像中展示。

读取**图片**和**音频**

```python
from tensorboard.summary.v1 import audio
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
from librosa import load

writer = SummaryWriter("logs/runs/test/2")
# Add images
image_path = "dataset/test1/135994133_4f306fe4bf_n.jpg"  # 注意这里的路径是从该py文件所在目录开始的
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)
writer.add_image("test_image", img_array, 0, dataformats="HWC")
# Add audio
audio_path = "dataset/test1/p226_002.wav"
audio_data, sr = load(audio_path)
writer.add_audio("test_audio", audio_data, 0, sample_rate=sr)

writer.close()
```



#### **打开方式**

终端使用以下语句：

```bash
tensorboard --logdir=tensorboard/logs/runs/test/1							# logdir=事件文件所在文件夹名
```

调整服务器端口号

```bash
tensorboard --logdir=tensorboard/logs/runs/test/1 --port=6007				# 调整服务器号为6007
```



