# Mossformer模型总体架构
采用的是时域分离的方法，即 **编码** ——> **掩码** ——> **解码** 的流程
![[Pasted image 20250508214334.png|600]]
输入序列$\mathbf{X}\in\mathbb{R}^{B\times1\times T}$
#Encode
核大小为$K_1$，步长为$K_1/2$，卷积核(滤波器)的数量是$N$，
$\mathbf{X}^{^{\prime}}=\mathrm{ReLU}(\mathrm{Conv}1\mathrm{D}(\mathbf{X}))$，其中 $\mathbf{X}^{^{\prime}}\in\mathbb{R}^{B\times N\times S}$ , $S=2(T-K_{1})/K_{1}+1$
将序列 $𝐗^′$ 逐个元素乘以每个说话人的掩码，得到分离的特征序列：$\mathbf{X}_i^{\prime\prime}=\mathbf{M}_i\otimes\mathbf{X^{\prime}}$
#Decode
解码器是一维转置卷积层，它使用与编码器相同的核大小和步幅。
最终，解i码器将分离的特征序列解码为波形:$\hat{s}_i=\text{Transposed Conv}1\mathrm{D}(\mathbf{X}_i^{\prime\prime})$
#Masking掩蔽
掩蔽网络执行从编码器输出到 $C$ 组掩蔽的非线性映射(掩蔽的输出为C个不同批次的掩码)，为了分离获得 $C$ 组源信号
首先对编码序列 $𝐗^′$ 进行归一化，并添加位置编码以获取全局顺序信息。然后，该序列经过逐点卷积，并在重塑后传递至 R 个 MossFormer 模块进行顺序处理。


# Mossformer模块
## 带有卷积增强联合自注意力的门控单头 Transformer（ GSHT ）架构
即gated single-head transformer (GSHT) architecture with convolution-augmented joint self-attentions
GSHT 架构采用了强大的注意力门控机制，因此只需要弱化版的单头自注意力（single-head self-attention (SHSA)）。
### 在 MossFormer 模块中，序列由卷积模块和注意力门控机制处理。
卷积模块使用线性投影和深度卷积处理序列。
注意力门控机制执行局部和全局的联合自注意力和门控操作。
MossFormer 模块仅**学习残差**(残差学习是指神经网络不直接学习目标输出 Y，而是*学习输入 X 与目标输出 Y 之间的差值*（残差）)，并应用来自输入的跳跃连接，以便于训练。当前 MossFormer 模块的输出将作为下一个 MossFormer 模块的输入。MossFormer 模块的流程重复 R 次。
最终 MossFormer 模块的输出经过 ReLU 处理(因掩码用于**幅度调整**，物理意义上需非负)，随后进行另一次[[PW逐点卷积Pointwise Conv]]，将序列维度从 $\mathbb{R}^{N\times S}$ 到 $\mathbb{R}^{(C\times N)\times S}$ 。
掩码序列 $𝐌$ 会针对每个说话者 $\mathbf{M}_{i}\in\mathbb{R}^{N\times S}$ 进行重组，一个说话人对应一串掩码，然后分别输入解码器。实现从混合音频中得到目标扬声器的特征。

#### 卷积模块
该模块基于近期提出的用于长序列建模的门控注意力单元(GAU)[[12](https://ar5iv.labs.arxiv.org/html/2202.10447?immersive_translate_auto_translate=1)]开发而成。提出了一个卷积模块来取代 GAU 中的密集层，用于在 MossFormer 模块中提取细粒度的局部特征模式。
为了进一步建模按位置(位置是从注意力机制中的位置编码中获取的信息)的局部特征模式，使用了下面的卷积模块![[Pasted image 20250509194242.png|800]]
通过整合卷积模块和三重门控结构来提升 MossFormer 模块的建模能力。使用门控可以使 SHSA 更加简单，从而促进联合局部和全局注意力，从而实现有效的长距离建模。

#### 注意力门控机制
注意力门控机制将注意力机制融入三重门控过程中，以增强模型能力。
![[Pasted image 20250509193711.png|500]]

##### 联合局部和全局的单头自注意力
传统双路径 Transformer（如 SepFormer）的全局注意力受限于二次复杂度（$O(S^2)$）,效率低下。
##### 实现流程
令当前Mossformer模块的输入序列为 $\mathbf{X}^{\prime\prime}\in\mathbb{R}^{B \times S\times N}$，该序列经*卷积模块* ( $\mathbf{U=ConvM(X^{\prime\prime})},\quad\mathbf{V=ConvM(X^{\prime\prime})}$ )处理后，得到 $\mathbf{U}\in\mathbb{R}^{B \times S\times2N}$ 和 $\mathbf{V}\in\mathbb{R}^{B \times S\times2N}$
卷积模块通过扩展因子为2的线性层，将特征维度从 $N$ 升维到 $2N$。
通过联合局部和全局的注意力机制
$$\begin{aligned}&\mathbf{O^{\prime}}=\phi(\mathbf{U}\otimes\mathbf{V^{\prime}})\quad\mathrm{where}\quad\mathbf{V^{\prime}}=\mathbf{AV}\\&\mathbf{O^{\prime\prime}}=\mathbf{U^{\prime}}\otimes\mathbf{V}\quad\mathrm{where}\quad\mathbf{U^{\prime}}=\mathbf{AU}\\&\mathbf{O}=\mathbf{X}^{\prime\prime}+\mathrm{Conv}\mathbf{M}(\mathbf{O}^{\prime}\otimes\mathbf{O}^{\prime\prime})\end{aligned}$$ 
但是对于对于 $S$ 很大的长序列，直接计算上面的 $O^{\prime}$和 $O^{{\prime}{\prime}}$公式所需要的注意力机制成本很高，于是采用了[[GAU门控注意力单元]]中的门控机制高效地联合局部与全局注意力，实现较低成本计算 $V^{\prime}$ 和 $U^{\prime}$。具体流程如下： ^e4fc62


## MossFormer 的**联合注意力**旨在解决两个核心问题：
### 1. 长序列计算效率：
避免直接计算全序列二次注意力（如 SepFormer 的多头注意力），通过分块局部 + 线性全局的混合模式，将复杂度降至$O(S \cdot P + S \cdot D)$（P 为块大小，D 为注意力维度）。
### 2. 直接长距离交互：
打破双路径框架的块间间接依赖（如通过 RNN 传递状态），使任意两帧可通过全局注意力直接交互，同时保留局部精细建模。
直接计算$O^{\prime}$和$O^{\prime\prime}$其效率很低，门控的存在使我们能够基于联合局部和全局注意力机制高效地计算 $𝐕^′$ 和 $𝐔^′$ 。
- 我们首先通过卷积模块将输入序列$X^{\prime\prime}\in\mathbb{R}^{B \times S\times N}\text{(B为批次,S 为序列长度,N 为特征维度)}$ 投影到共享表示中：$\mathbf{Z}=\mathrm{ConvM}(\mathbf{X}^{\prime\prime})\in\mathbb{R}^{B \times S\times D}$ ，其中 $D≪N$ 。注意图2中的$\mathbf{Z}\in\mathbb{R}^{B \times S\times A}$应该是错的，应该是 $\mathbf{Z}\in\mathbb{R}^{B \times S\times D}$ ，$A$是注意力矩阵$\mathbf{A}\in\mathbb{R}^{S\times S}$ 
- 然后，我们将低成本的每维标量和偏移量以及 RoPE(旋转位置编码) [13](https://ar5iv.labs.arxiv.org/html/2302.11824?immersive_translate_auto_translate=1#bib.bib13)应用于共享的 $𝐙$ ，以获得局部和全局注意力机制的查询$\mathrm{Q}\mathrm{、Q}^{^{\prime}}\in\mathbb{R}^{B \times S\times D}$ 和键$\mathrm{K、K}^{^{\prime}}\in\mathbb{R}^{B \times S\times D}$。[[GAU门控注意力单元#^55baef|Q、K与Z的关系]]
- 对于全局注意力机制，我们采用以下低成本线性化形式来捕捉序列 𝐕 和 𝐔 的长距离全局交互：$$\mathbf{V}_{\mathrm{global}}^{\prime}=\mathbf{Q}^{\prime}\left(\beta\mathbf{K}^{\prime T}\mathbf{V}\right),\quad\mathbf{U}_{\mathrm{global}}^{\prime}=\mathbf{Q}^{\prime}\left(\beta\mathbf{K}^{\prime T}\mathbf{U}\right)$$其中 $β=1/S$ 是缩放因子。长距离全局交互没有使用![[Multi-Head Self-Attention#^572aa1]]$\mathbf{K}^{\prime T}\mathbf{V}$ 和 $\mathbf{K}^{\prime T}\mathbf{U}$ $\in\mathbb{R}^{B \times D\times 2N}$，$\mathbf{V}_{\mathrm{global}}^{\prime}$ 和 $\mathbf{U}_{\mathrm{global}}^{\prime}$ $\in\mathbb{R}^{B \times S\times 2N}$
为了计算局部二次注意力机制，我们将 $𝐕$ 、 $𝐔$ 、 $𝐐$ 和 $𝐊$ 分成 $H$ 个大小为 $P$ 的互不重叠的块，其中 $S<H×P$ 时使用零填充。$\mathbf{U}_h\in\mathbb{R}^{B \times S\times2N/P}$ 和 $\mathbf{V}_h\in\mathbb{R}^{B \times S\times2N/P}$ ，$\mathrm{Q}_h\in\mathbb{R}^{B \times S\times D/P}$ 和键$\mathrm{K}_h\in\mathbb{R}^{B \times S\times D/P}$因此，二次注意力机制会独立地应用于每个块，如下所示：$$\mathbf{V}_{\mathrm{local},h}^{^{\prime}}=\mathrm{ReLU}^2\left(\gamma\mathbf{Q}_h\mathbf{K}_h^T\right)\mathbf{V}_h,\mathbf{U}_{\mathrm{local},h}^{^{\prime}}=\mathrm{ReLU}^2\left(\gamma\mathbf{Q}_h\mathbf{K}_h^T\right)\mathbf{U}_h$$其中 $γ=1/P$ 是缩放因子。$\mathbf{Q}_h\mathbf{K}_h^T$为了优化性能，我们采用平方 $ReLU$ 代替 $MHSA$ 中的 $softmax$ [12](https://ar5iv.labs.arxiv.org/html/2302.11824?immersive_translate_auto_translate=1#bib.bib12)  。注意， $𝐐_h​𝐊_h^T$ 只需计算一次，因为它由 $𝐕_{{local},h^′}$ 和 $𝐔_{{local},h^′}$ 共享。我们将 $𝐕^′_{{local},h}$ 和 $𝐔^′_{{local},h}$ 的所有输出沿时间维度连接起来，形成完整序列： $𝐕_{local}^′=[𝐕^′_{{local},1},…,𝐕^′_{{local},H}]$ 和 $𝐔_{local}^′=[𝐔^′_{{local},1},…,𝐔^′_{{local},H}]$。局部注意力和全局注意力加在一起，形成 [[Mossformer模型#^e4fc62|公式内]] 最终的联合注意力 $𝐕^′$ 和 $𝐔^′$ ：$$\mathbf{V}^{^{\prime}}=\mathbf{V}_{\mathrm{local}}^{^{\prime}}+\mathbf{V}_{\mathrm{global}}^{^{\prime}},\quad\mathbf{U}^{^{\prime}}=\mathbf{U}_{\mathrm{local}}^{^{\prime}}+\mathbf{U}_{\mathrm{global}}^{^{\prime}}$$


 