VirtualBatchNorm1d（虚拟批量归一化）是 Batch Normalization（BN）的改进变体，主要针对**小批量（small batch size）训练场景**设计，其核心作用和优势如下：
## 核心作用

1. **归一化数据分布** 与标准 BN 相同，通过对输入数据进行均值和方差归一化，加速模型收敛并提升泛化能力。具体对 1D 数据（如序列特征、全连接层输出），按特征维度（channel）计算均值和方差：$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} \cdot \gamma + \beta$ 其中 $\mu_B$,$\sigma_B^2$ 是当前批次的统计量，$\gamma$, $\beta$ 是可学习参数。
2. **模拟更大批量统计量** 在小批量场景下，引入**虚拟批量**概念：保存一个 “参考批量”（如首次输入的固定大小数据），后续训练时始终使用该参考批量的均值和方差进行归一化，而非当前小批量的统计量。

## 核心优势
1. 缓解**小批量**训练的**统计噪声**
    标准 BN 依赖当前批次的统计量，当 batch size 较小时（如 1-8），均值和方差估计波动大，导致训练不稳定、模型性能下降。
    VBN 通过固定参考批量（如预设的较大批量，如 128），用其统计量替代当前小批量的统计量，避免小批量噪声对归一化的影响，在内存受限场景（如图像生成、资源有限的设备）中优势显著。
2. **减少**对**批量大小的依赖**
    训练和推理时无需适配不同批量大小，尤其适合动态批量场景（如在线学习、流式数据）。
    传统 BN 在推理时需依赖训练时的全局统计量（均值 / 方差的移动平均），而 VBN 的参考批量统计量固定，可直接用于推理，简化部署流程。
3. 稳定训练初期的归一化效果
    训练初期，标准 BN 的统计量尚未收敛，可能导致梯度消失 / 爆炸。VBN 使用预设的参考批量（通常基于初始数据计算），初始归一化更稳定，加速模型初始化。
4. **适用特定任务场景**
    尤其在生成对抗网络（GANs）中，生成器通常使用小批量训练以避免模式崩溃，VBN 可提升生成样本的质量和训练稳定性。
    对内存敏感的任务（如多 GPU 分布式训练中单个 GPU 的小批量），VBN 在保持性能的同时降低显存占用。
## 与标准 BN 的对比

|特性|标准 BN|VirtualBatchNorm1d|
|---|---|---|
|统计量来源|当前批次|固定参考批次|
|适合批量大小|较大（如 32+）|小批量（如 1-16）|
|训练稳定性|小批量时波动大|小批量时更稳定|
|推理依赖|训练时的移动平均|固定参考批量的统计量|

### 总结
VirtualBatchNorm1d 通过引入固定参考批量的统计量，解决了标准 BN 在小批量训练中的核心缺陷，在内存受限、动态批量或对统计噪声敏感的场景中具有显著优势。其核心价值是**在小批量下保持归一化的稳定性和有效性**，拓宽了批量归一化的适用范围。


```python
class VirtualBatchNorm1d(Module):
    """
    Module for Virtual Batch Normalization.
    Implementation borrowed and modified from Rafael_Valle's code + help of SimonW from this discussion thread:
    https://discuss.pytorch.org/t/parameter-grad-of-conv-weight-is-none-after-virtual-batch-normalization/9036
    """
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        # batch statistics
        self.num_features = num_features
        self.eps = eps  # epsilon
        # define gamma and beta parameters
        self.gamma = Parameter(torch.normal(mean=1.0, std=0.02, size=(1, num_features, 1)))
        self.beta = Parameter(torch.zeros(1, num_features, 1))
  
    def get_stats(self, x):
        """
        Calculates mean and mean square for given batch x.
        Args:
            x: tensor containing batch of activations
        Returns:
            mean: mean tensor over features
            mean_sq: squared mean tensor over features
        """
        mean = x.mean(2, keepdim=True).mean(0, keepdim=True)
        mean_sq = (x ** 2).mean(2, keepdim=True).mean(0, keepdim=True)
        return mean, mean_sq
  
    def forward(self, x, ref_mean, ref_mean_sq):
        """
        Forward pass of virtual batch normalization.
        Virtual batch normalization require two forward passes
        for reference batch and train batch, respectively.
        Args:
            x: input tensor
            ref_mean: reference mean tensor over features
            ref_mean_sq: reference squared mean tensor over features
        Result:
            x: normalized batch tensor
            ref_mean: reference mean tensor over features
            ref_mean_sq: reference squared mean tensor over features
        """
        mean, mean_sq = self.get_stats(x)
        if ref_mean is None or ref_mean_sq is None:
            # reference mode - works just like batch norm
            mean = mean.clone().detach()
            mean_sq = mean_sq.clone().detach()
            out = self.normalize(x, mean, mean_sq)
        else:
            # calculate new mean and mean_sq
            batch_size = x.size(0)
            new_coeff = 1. / (batch_size + 1.)
            old_coeff = 1. - new_coeff
            mean = new_coeff * mean + old_coeff * ref_mean
            mean_sq = new_coeff * mean_sq + old_coeff * ref_mean_sq
            out = self.normalize(x, mean, mean_sq)
        return out, mean, mean_sq

    def normalize(self, x, mean, mean_sq):
        """
        Normalize tensor x given the statistics.
        Args:
            x: input tensor
            mean: mean over features
            mean_sq: squared means over features
        Result:
            x: normalized batch tensor
        """
        assert mean_sq is not None
        assert mean is not None
        assert len(x.size()) == 3  # specific for 1d VBN
        if mean.size(1) != self.num_features:
            raise Exception('Mean tensor size not equal to number of features : given {}, expected {}'
                            .format(mean.size(1), self.num_features))
        if mean_sq.size(1) != self.num_features:
            raise Exception('Squared mean tensor size not equal to number of features : given {}, expected {}'
                            .format(mean_sq.size(1), self.num_features))
        std = torch.sqrt(self.eps + mean_sq - mean ** 2)
        x = x - mean
        x = x / std
        x = x * self.gamma
        x = x + self.beta
        return x
    def __repr__(self):
        return ('{name}(num_features={num_features}, eps={eps}'
                .format(name=self.__class__.__name__, **self.__dict__))
```

### 使用方式
- 事先将参考批量数据(随机获取的)放入模型中，并记录其参考批次的各个层的均值和平方均值，计算完毕后，再走正常的前向预测部分。有以下三个部分的难点：
    - 获取参考批次
    - 记录参考批次的各层均值和平方均值
    - 前向预测
#### 获取参考批次
可以在数据集函数中定义下面的方法
```python
    def ref_batch(self,batch_size):
        index = np.random.choice(len(self.clean_files),batch_size).tolist()  # 随机选择batch_size个索引
        catch_clean = [emphasis(np.load(self.clean_files[i])) for i in index]
        catch_noisy = [emphasis(np.load(self.noisy_files[i])) for i in index]
        catch_clean = np.expand_dims(np.array(catch_clean),axis=1)
        catch_noisy = np.expand_dims(np.array(catch_noisy),axis=1)
        batch_wav = np.concatenate((catch_clean,catch_noisy),axis=1)
        return torch.from_numpy(batch_wav)
```
#### 记录参考值
```python
    def forward(self, x, ref_x):
        """
        Forward pass of discriminator.
        Args:
            x: input batch (signal)
            ref_x: reference input batch for virtual batch norm
        """
        # reference pass
        ref_x = self.conv1(ref_x)
        ref_x, mean1, meansq1 = self.vbn1(ref_x, None, None)
        ref_x = self.lrelu1(ref_x)
        ref_x = self.conv2(ref_x)
        ref_x, mean2, meansq2 = self.vbn2(ref_x, None, None)
        ref_x = self.lrelu2(ref_x)
        ref_x = self.conv3(ref_x)
        ref_x = self.dropout1(ref_x)
        ref_x, mean3, meansq3 = self.vbn3(ref_x, None, None)
        ref_x = self.lrelu3(ref_x)
        ref_x = self.conv4(ref_x)
        ref_x, mean4, meansq4 = self.vbn4(ref_x, None, None)
        ref_x = self.lrelu4(ref_x)
        ref_x = self.conv5(ref_x)
        ref_x, mean5, meansq5 = self.vbn5(ref_x, None, None)
        ref_x = self.lrelu5(ref_x)
        ref_x = self.conv6(ref_x)
        ref_x = self.dropout2(ref_x)
        ref_x, mean6, meansq6 = self.vbn6(ref_x, None, None)
        ref_x = self.lrelu6(ref_x)
        ref_x = self.conv7(ref_x)
        ref_x, mean7, meansq7 = self.vbn7(ref_x, None, None)
        ref_x = self.lrelu7(ref_x)
        ref_x = self.conv8(ref_x)
        ref_x, mean8, meansq8 = self.vbn8(ref_x, None, None)
        ref_x = self.lrelu8(ref_x)
        ref_x = self.conv9(ref_x)
        ref_x = self.dropout3(ref_x)
        ref_x, mean9, meansq9 = self.vbn9(ref_x, None, None)
        ref_x = self.lrelu9(ref_x)
        ref_x = self.conv10(ref_x)
        ref_x, mean10, meansq10 = self.vbn10(ref_x, None, None)
        ref_x = self.lrelu10(ref_x)
        ref_x = self.conv11(ref_x)
        ref_x, mean11, meansq11 = self.vbn11(ref_x, None, None)
```
#### 前向预测
```python
        x = self.conv1(x)
        x, _, _ = self.vbn1(x, mean1, meansq1)
        x = self.lrelu1(x)
        x = self.conv2(x)
        x, _, _ = self.vbn2(x, mean2, meansq2)
        x = self.lrelu2(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        x, _, _ = self.vbn3(x, mean3, meansq3)
        x = self.lrelu3(x)
        x = self.conv4(x)
        x, _, _ = self.vbn4(x, mean4, meansq4)
        x = self.lrelu4(x)
        x = self.conv5(x)
        x, _, _ = self.vbn5(x, mean5, meansq5)
        x = self.lrelu5(x)
        x = self.conv6(x)
        x = self.dropout2(x)
        x, _, _ = self.vbn6(x, mean6, meansq6)
        x = self.lrelu6(x)
        x = self.conv7(x)
        x, _, _ = self.vbn7(x, mean7, meansq7)
        x = self.lrelu7(x)
        x = self.conv8(x)
        x, _, _ = self.vbn8(x, mean8, meansq8)
        x = self.lrelu8(x)
        x = self.conv9(x)
        x = self.dropout3(x)
        x, _, _ = self.vbn9(x, mean9, meansq9)
        x = self.lrelu9(x)
        x = self.conv10(x)
        x, _, _ = self.vbn10(x, mean10, meansq10)
        x = self.lrelu10(x)
        x = self.conv11(x)
        x, _, _ = self.vbn11(x, mean11, meansq11)
        x = self.lrelu11(x)
        x = self.conv_final(x)
        x = self.lrelu_final(x)
        # reduce down to a scalar value
        x = torch.squeeze(x)
        x = self.fully_connected(x)
        return self.sigmoid(x)
```