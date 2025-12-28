# 视觉编码器与投影层

视觉语言模型 (VLM) 是多模态大模型的基础。虽然本知识库侧重于语音，但理解 VLM 的架构对于构建视听融合 (Audio-Visual) 模型至关重要。

## 1. 视觉编码器 (Vision Encoder)

负责将图像转换为特征向量。

### 1.1 ViT (Vision Transformer)
将图像切分为 Patch (如 16x16)，展平后作为 Transformer 的输入。

### 1.2 CLIP-ViT
使用对比学习预训练的 ViT，其特征空间与文本对齐。大多数 VLM (如 LLaVA, MiniGPT-4) 直接使用冻结的 CLIP-ViT 作为视觉编码器。

---

## 2. 模态对齐与投影 (Projection)

如何将视觉特征 $F_v$ 注入到 LLM 中？

### 2.1 Linear Projection
最简单的方法。
$$
E_v = W \cdot F_v + b
$$
LLaVA 使用此方法，证明了简单的线性映射足以保留丰富的视觉信息。

### 2.2 Q-Former (BLIP-2)
使用一组可学习的 Query Tokens，通过 Cross-Attention 从冻结的 ViT 中提取视觉特征。
*   **优点**: 压缩了视觉 Token 的数量（如从 256 个 Patch 压缩到 32 个 Query），降低了 LLM 的计算负担。

---

## 3. 视听融合 (Audio-Visual Integration)

在视频理解任务中，仅看画面是不够的（如判断视频中的人在说什么，或识别背景音乐）。

### 3.1 Video-LLaMA
同时引入 Vision-Language Branch 和 Audio-Language Branch。
*   **Vision Branch**: ViT -> Q-Former -> Linear -> LLM.
*   **Audio Branch**: ImageBind-Audio -> Linear -> LLM.
*   **联合理解**: LLM 接收 `[VISUAL_TOKENS]` 和 `[AUDIO_TOKENS]`，综合判断视频内容。

### 3.2 跨模态对齐的重要性
如果视觉看到“狗张嘴”，音频听到“汪汪叫”，模型应能关联这两个事件。ImageBind 提供了这种天然的对齐能力。
