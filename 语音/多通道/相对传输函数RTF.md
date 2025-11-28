相对传递函数（RTF, Relative Transfer Function）
# 1. 物理模型回顾

$\mathbf X_{f,t} = \mathbf c_f S_{f,t} + \mathbf r_f N_{f,t}$
维度：

| 符号                 | 含义              | 维度           |
| ------------------ | --------------- | ------------ |
| $\mathbf X_{f,t}$​ | P 通道的混合 STFT    | $P \times 1$ |
| $\mathbf c_f$      | 语音的 RTF         | $P \times 1$ |
| $S_{f,t}$          | 参考通道语音（complex） | 标量           |
| $\mathbf r_f$      | 噪声的 RTF         | $P \times 1$ |
| $N_{f,t}$​         | 参考通道噪声（complex） | 标量           |

也就是说，**每个通道的语音成分是参考通道语音的线性缩放 + 相位旋转**。

---

#  2. 什么是 RTF？(Relative Transfer Function)

RTF 是 **相对传递函数**，它描述了：**目标（语音或噪声）在不同麦克风之间的相对幅度和相位关系。**
换句话说：
- 每个声源到每个麦克风都有传递函数 $h_p(t)$（空间位置 + 房间混响决定）
- 但我们不关心声源的真实频谱，只关心**各个麦克风之间的相对到达特性**


因此：
### RTF 定义：
$c_f = \frac{H_f^{(speech)}}{ H_f^{(speech)}(ref) }$

1. 其中 $H_f^{(speech)}(ref)$是参考麦克风（通常是第 0 个）。
2. 于是 $c_f(0) = 1$（参考通道的相对传递函数定义为 1）。
物理意义：
- $c_f(p)$ 告诉你：**语音源对第 p 个麦克风的影响，相对参考麦克风的幅度比与相位差**。
同理，对噪声：
$\mathbf r_f = \frac{H_f^{(noise)}}{ H_f^{(noise)}(ref) }$

---

# 3. RTF 的物理意义（容易理解的解释）

对于语音源（假设直达占主导）：
$\mathbf c_f(p) \approx e^{-j 2\pi f \tau_{p}/F_s}\cdot \alpha_p$

包含两部分：
1. **相位项：**  
    与麦克风 p 相对参考通道的传播延迟 $\tau_p$ 决定 ——表现为空间信息（DoA）
2. **幅度项：**  
    $\alpha_p$ 是距离、遮挡、麦克风灵敏度导致的增益差
因此，RTF 是麦克风阵列中某个声源的“空间指纹”。

---

# 4. RTF 为什么重要？
因为传统 beamforming（如 MVDR、Gevd、LCMV）依赖它：
$\mathbf w = \frac{ \mathbf \Phi_{nn}^{-1} \mathbf c_f } { \mathbf c_f^H \mathbf \Phi_{nn}^{-1} \mathbf c_f }$
在不估计 RTF 时，波束形成方向无法确定，也无法实现无失真约束。
EaBNet 的创新点之一是：**不显式估计 RTF，而让 EM + BM 网络直接学习权重。**

---

# 5. 如何从物理模型理解 RTF？
从模型：
$\mathbf X_{f,t} = \mathbf c_f S_{f,t} + \mathbf r_f N_{f,t}$
第 p 个通道为：
$X_{p,f,t} = c_f(p) \cdot S_{f,t} + r_f(p) \cdot N_{f,t}$
但：
- **S_{f,t} 在所有麦克风是相同的（来自同一个声源）**
- 差异只来自多通道通路的差异 → 就是 RTF

**RTF 完全由“空间传播路径 + 混响 + 麦克风特性”决定，与信号内容无关。**

---

# 6. RTF 如何估计？（经典方法 vs 神经网络方法）

## （A）传统方法（基于 STFT）

### ① 直接用语音段做比值（简化型）：
当你有语音段的掩码（如来自 VAD 或网络）时：
$c_f(p) = \frac{ \sum_t M_{f,t}^{(speech)} X_{p,f,t} } { \sum_t M_{f,t}^{(speech)} X_{ref,f,t} }$
但这种仅适用于低混响环境。

---

### ② 基于 GEVD/SCM 的经典 RTF 估计
求解：
$\Phi_{xx} c_f = \lambda \Phi_{nn} c_f$
其中
- $\Phi_{xx}$ 是语音+噪声协方差矩阵
- $\Phi_{nn}$ 是噪声协方差矩阵
$c_f$ 就是 GEVD 最大特征向量（归一化后使 ref 为 1）。
这是 **MVDR 波束形成** 的经典做法。

---
### ③ 基于方向（DoA）推导的 RTF
若已知道声源方位 $\theta$：
$c_f(p) = e^{-j 2 \pi f d_p(\theta)/c }$
其中 d_p 是参考通道到第 p 通道的路径差。  
这适用于近似自由场（少混响）环境。

---

## （B）深度学习方法（RTF-free Beamformers）
如本论文 EaBNet（embedding + neural beamforming）：不再显式估计 RTF，而是直接学习 beamforming 权重。
但其隐含学习到的空间模式 **等价于 RTF（但更鲁棒）**。

---

# 7. 总结：RTF 是什么？如何理解？

### 定义
RTF（相对传递函数）是：多通道系统中某声源从参考麦克风到其他麦克风的传递函数比值（复数形式）。

---

### 物理意义
包含麦克风之间的 **相位差 + 幅度差**，描述空间结构（声源位置 + 室内反射）。

---

### 与声源信号内容无关
它由传播路径决定，与语音/噪声的真实谱无关。

---
### 在传统 beamforming 中非常关键
用来确定“无失真方向”或“抑制方向”。
