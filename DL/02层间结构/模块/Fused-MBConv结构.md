Fused-MBConv 结构由 **EfficientNetV2**（2021, Tan & Le）提出。

---

## 一、背景：DW 卷积的实际效率问题

**深度可分离卷积（DW Conv）** 理论计算量低，但实际运行效率存在问题：

| **问题**         | **说明**                             |
| ---------------- | ------------------------------------ |
| **硬件利用率低** | DW 卷积无法充分利用 GPU/TPU 的并行计算能力 |
| **内存访问开销大** | 需要分别执行 DW 和 PW 卷积，内存带宽成为瓶颈 |
| **实际速度不理想** | 理论 FLOPs 低，但实际推理速度未必快     |

---

## 二、核心思想

将 MBConv 中的 **1×1 升维卷积 + DW 卷积** 融合为一个 **3×3 标准卷积**，在浅层网络中提升实际效率。

---

## 三、MBConv vs Fused-MBConv

<img src="../../assets/image-20241119215204547.png" alt="MBConv与Fused-MBConv对比" style="zoom: 50%; display: block; margin: 0 auto;" />

### 3.1 MBConv（原始）

```
输入 (C_in)
  │
  ├─ 1×1 Conv (升维 → C_in × t) → BN → Swish
  ├─ 3×3 DW Conv → BN → Swish
  ├─ [SE 模块]（可选）
  └─ 1×1 Conv (降维 → C_out) → BN
  │
  └─ [捷径分支]（stride=1 且 C_in=C_out 时）
```

### 3.2 Fused-MBConv

```
输入 (C_in)
  │
  ├─ 3×3 Conv (升维 → C_in × t) → BN → Swish    ← 融合升维和空间卷积
  ├─ [SE 模块]（可选，实际常省略）
  └─ 1×1 Conv (降维 → C_out) → BN（若 t=1 则无此层）
  │
  └─ [捷径分支]（stride=1 且 C_in=C_out 时）
```

---

## 四、计算量对比

设输入通道 $C$，扩展因子 $t$，空间尺寸 $H \times W$：

| **结构**       | **计算量（MACs）**                                   |
| -------------- | ---------------------------------------------------- |
| MBConv         | $HW \cdot C \cdot (tC + 9t + tC_{out})$              |
| Fused-MBConv   | $HW \cdot C \cdot (9tC + tC_{out})$                  |

**分析**：
- 当 $C$ 较小（浅层）时，Fused-MBConv 的额外计算量可接受，但硬件效率更高
- 当 $C$ 较大（深层）时，MBConv 的计算量优势明显

---

## 五、EfficientNetV2 的混合策略

| **网络位置** | **推荐结构**   | **原因**                           |
| ------------ | -------------- | ---------------------------------- |
| 浅层（1-3）  | Fused-MBConv   | 通道数少，标准卷积硬件效率高       |
| 深层（4+）   | MBConv         | 通道数多，DW 卷积计算量优势明显    |

---

## 六、特殊情况：扩展因子 t=1

当扩展因子 $t=1$ 时，Fused-MBConv 简化为单个 3×3 卷积：

<img src="../../assets/image-20241119225931388.png" alt="Fused-MBConv实际结构" style="zoom: 50%; display: block; margin: 0 auto;" />

```
输入 (C_in)
  │
  ├─ 3×3 Conv → BN → Swish    ← 无通道扩展
  │
  └─ [捷径分支]（stride=1 且 C_in=C_out 时）
```

---

## 七、SE 模块的使用

| **场景**               | **SE 模块**        |
| ---------------------- | ------------------ |
| 原始 EfficientNet      | 使用               |
| EfficientNetV2 浅层    | **通常省略**       |
| EfficientNetV2 深层    | 使用               |

> **原因**：浅层特征图较大，SE 模块的全局池化开销相对较高；深层特征图小，SE 带来的性能提升更明显。
