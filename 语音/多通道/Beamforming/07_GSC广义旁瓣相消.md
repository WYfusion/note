# GSC 广义旁瓣相消器

## 1. 概述

**GSC (Generalized Sidelobe Canceller)** 是一种自适应波束形成结构，由 Griffiths 和 Jim 于 1982 年提出。它将约束优化问题转化为无约束优化问题，使得自适应滤波器的设计更加简单。

### 1.1 核心思想

将波束形成器分解为两条路径：
1. **固定路径**：保证目标信号无失真通过
2. **自适应路径**：估计并消除残余噪声和干扰

### 1.2 与MVDR的关系

GSC是MVDR的一种等价实现，但结构更清晰，更易于实现和分析。

---

## 2. GSC结构

### 2.1 系统框图

```
输入 X(f,t)
    ↓
    ├─────────────────────┐
    ↓                     ↓
[固定波束形成器]      [阻塞矩阵]
    w_q                   B
    ↓                     ↓
  Z(f,t)               U(f,t)
    ↓                     ↓
    |                [自适应滤波器]
    |                     w_a
    |                     ↓
    └──────(-)───────── Y_a(f,t)
           ↓
        Y(f,t)
```

**数学表达**：
$$Y(f,t) = Z(f,t) - Y_a(f,t)$$

其中：
- $Z(f,t) = \mathbf{w}_q^H \mathbf{X}(f,t)$：固定波束形成器输出
- $\mathbf{U}(f,t) = \mathbf{B}^H \mathbf{X}(f,t)$：阻塞矩阵输出
- $Y_a(f,t) = \mathbf{w}_a^H \mathbf{U}(f,t)$：自适应滤波器输出

### 2.2 完整表达式

$$Y(f,t) = (\mathbf{w}_q - \mathbf{B}\mathbf{w}_a)^H \mathbf{X}(f,t)$$

等价的波束形成权重：
$$\mathbf{w}_{\text{GSC}} = \mathbf{w}_