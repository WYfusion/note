主要使用了[[通道重排（Channel Shuffle）#^a2e53a|通道重排(shuffle)]]和[[分组卷积#^d8aaf1|分组卷积]]或[[PW逐点卷积Pointwise Conv#^9be128|PW卷积]]


## shuffleNet-v1

### 结构块构成

其中的(a)为对照结构，不是shuffleNet的结构

![image-20241113220608569](image-20241113220608569.png)

- (a)是ResNeXt中的结构，PW卷积(1×1的卷积)占据了93.4%的理论计算量。
- (b)是shuffleNet-v1中的结构，将PW卷积换成了分组卷积，加入了通道重排，均采用步距为1，是采用AddTensor，通道数不变
- (c)是shuffleNet-v1中的结构，使用平均池化层对于分支1进行下采样，最后使用的是Concat拼接通道，步距为2，涉及了尺寸的减半和通道数加倍

## shuffleNet-v2

### 结构块构成

其中的(a)(b)是shuffleNet-v1的结构，目的是作为对照

为了满足shuffleNet-v2对应论文中所提出的关于模型的4个设计指标要求，将分组卷积改成了1×1的卷积（PW逐点卷积）

<img src="../assets/image-20241114091953527.png" alt="image-20241114091953527" style="zoom:50%;" />

由于没有了分组卷积的存在，所以将1×1分组卷积后的Channel Shuffle移动到Concat后面

(c)图中步距为1经过通道分割， 再拼接后的通道数不变，尺寸也不变

(d)图中没有经过通道分割，再拼接后的通道数是翻倍的；同时含有步距为2的卷积块，尺寸减半

<img src="../assets/image-20241114175029474.png" alt="image-20241114175029474" style="zoom: 50%;" />