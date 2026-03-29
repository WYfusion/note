---
tags:
  - LLM/架构
  - LLM/多模态
created: 2026-03-29
updated: 2026-03-29
---

# ViT 如何把 Transformer 带进视觉

## 问题定义

图像并不是天然的 token 序列，ViT 是如何把它接入 Transformer 的？

## 直觉解释

ViT 的关键动作是把图像切成 patch，把每个 patch 当成一个 token。

## 形式化推导

图像先被切成固定大小 patch，再线性投影为 patch embedding，外加位置编码后送入 encoder 堆叠。

## 工程意义

这说明 Transformer 并不只属于文本，只要输入能被 token 化，就能进入类似骨架。

## 常见误解

> [!warning] 常见误解
> - “ViT 只是把 BERT 直接搬到图像。” 过于粗糙，视觉里的 patch、分辨率和 2D 位置都很关键。

## 例子或反例

图像分辨率改变时，位置 embedding 常需要插值，这正体现了视觉 token 化和位置设计的特殊性。

## 相关链接

- [[01_BERT式双向编码器适合什么任务|BERT 式双向编码器适合什么任务]]
- [[05_跨模态Transformer_语音_视觉_视频的结构适配|跨模态 Transformer]]
- [[01_绝对位置编码_Sinusoidal_Learnable|绝对位置编码]]
