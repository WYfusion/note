## 概述

后训练量化（Post-Training Quantization, PTQ）无需重新训练即可将模型压缩至低精度。本页对比四大主流方案。

---

## 方案总览

|方案|年份|量化对象|精度|核心思路|是否需要校准数据|
|---|---|---|---|---|---|
|**[LLM.int](http://LLM.int)8()**|2022|权重 + 激活|INT8（混合）|异常值通道 FP16，其余 INT8|否|
|**SmoothQuant**|2022|权重 + 激活|W8A8|激活难度迁移到权重|是（少量）|
|**GPTQ**|2022|权重|3/4-bit|逐列量化 + 海森矩阵补偿|是（~128 样本）|
|**AWQ**|2023|权重|4-bit|保护显著权重通道|是（少量）|

---

## [LLM.int](http://LLM.int)8()：混合分解量化

### 核心发现

LLM 的激活中存在 **异常值通道**（outlier features）——少量维度的激活值远大于其他维度（可达 100x+）。

### 方法

1. 检测激活中的异常值维度（阈值 $alpha$，通常 6.0）

1. 异常值维度保持 FP16 矩阵乘

1. 其余维度使用 INT8 矩阵乘

1. 结果合并

$$Y = X_{outlier} \cdot W_{outlier}^{FP16} + \text{dequant}(X_{normal}^{INT8} \cdot W_{normal}^{INT8})$$

> [!important]
> 
> **优势**：无需校准数据，开箱即用。**劣势**：混合精度 kernel 效率不如纯 INT8，speedup 有限。

---

## SmoothQuant：W8A8 的桥梁

### 核心思路

激活量化难（异常值）、权重量化易 → 将激活的量化难度 **数学等价地迁移到权重**。

### 数学原理

$$Y = X \cdot W = (X \cdot \text{diag}(s)^{-1}) \cdot (\text{diag}(s) \cdot W)$$

- $s$ 是逐通道缩放因子

- $hat{X} = X / s$：激活变平滑（范围缩小）

- $hat{W} = s cdot W$：权重吸收难度（范围略增）

$$s_j = \frac{\max|X_j|^\alpha}{\max|W_j|^{1-\alpha}}, \quad \alpha \in [0.5, 0.75]$$

> [!important]
> 
> **关键**：$alpha$ 控制难度迁移比例。$alpha = 0.5$ 时权重和激活各承担一半量化难度。

---

## GPTQ：高质量低比特权重量化

### 核心思路

基于 **Optimal Brain Quantization（OBQ）** 的改进，利用海森矩阵信息逐列量化权重，同时补偿量化误差到未量化列。

### 算法流程

1. 计算近似海森矩阵 $H = 2X^TX$（使用校准数据）

1. **逐列处理**：
    
    - 量化当前列 $w_q = \text{quant}(w)$
    
    - 计算误差 $\delta = w - w_q$
    
    - 将误差补偿到剩余未量化列：$W_{:,j+1:} += delta cdot H_{q,j+1:} / H_{q,q}$
    

1. 可选：行内分组（group quantization），每 $g$ 个权重共享一个 scale/zero-point

### Python 伪代码

```Python
def gptq_quantize(W, H, n_bits=4, group_size=128):
    """
    GPTQ 逐列量化核心逻辑（简化版）
    W: [out_features, in_features] 权重矩阵
    H: [in_features, in_features] 海森矩阵
    """
    import torch
    n_out, n_in = W.shape
    Q = torch.zeros_like(W)
    
    # Cholesky 分解加速
    H_inv = torch.linalg.cholesky(H)
    
    for col in range(n_in):
        w = W[:, col].clone()
        d = H_inv[col, col]  # 对角元素
        
        # 量化当前列
        q = quantize_to_nbit(w, n_bits, group_size)
        Q[:, col] = q
        
        # 误差补偿到后续列
        err = (w - q) / d
        W[:, col+1:] -= err.unsqueeze(1) * H_inv[col, col+1:].unsqueeze(0)
    
    return Q
```

> [!important]
> 
> **GPTQ 优势**：4-bit 量化质量极高，接近 FP16 基线。**常见搭配**：group_size=128，4-bit，生态成熟（AutoGPTQ、ExLlama）。

---

## AWQ：保护显著通道

### 核心发现

不是所有权重通道同等重要——与**大激活值通道**对应的权重对精度影响最大。

### 方法

1. 用校准数据统计每个权重通道的激活重要性 $s_j = \max|X_{:,j}|$

1. 对重要通道做 per-channel scaling 保护精度

1. 然后再做标准量化

$$Q(w \cdot s) / s \approx w \quad \text{（重要通道量化误差更小）}$$

> [!important]
> 
> **AWQ vs GPTQ**：AWQ 更快（不需要逐列迭代），工程友好；GPTQ 理论质量略高。实际差异在 4-bit 时通常很小。

---

## 质量对比

|方案|精度|LLaMA-2-7B PPL 退化|量化速度|推理 speedup|生态成熟度|
|---|---|---|---|---|---|
|BF16 基线|16-bit|-|-|1x|-|
|[LLM.int](http://LLM.int)8()|8-bit|~0%|快|~1.0-1.2x|⭐⭐⭐|
|SmoothQuant|W8A8|~0.1%|快|~1.5x|⭐⭐⭐|
|GPTQ 4-bit|4-bit|~0.3-0.5%|慢（分钟级）|~2-3x|⭐⭐⭐⭐|
|AWQ 4-bit|4-bit|~0.3-0.5%|快|~2-3x|⭐⭐⭐⭐|