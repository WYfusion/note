# 从训练到上线：Release 流程

MLOps (Machine Learning Operations) 旨在实现机器学习模型的标准化、自动化交付。对于语音大模型，由于数据体积大、实时性要求高，其工程链路具有特殊性。

## 1. 标准 MLOps 闭环

$$
\text{Data} \rightarrow \text{Training} \rightarrow \text{Evaluation} \rightarrow \text{Deployment} \rightarrow \text{Monitoring} \rightarrow \text{Data Loop}
$$

---

## 2. 语音数据工程 (Audio Data Ops)

语音数据的处理比文本更消耗资源。

### 2.1 预处理 (Preprocessing)
*   **重采样 (Resampling)**: 统一采样率（如 Whisper 使用 16kHz，MusicGen 使用 32kHz）。
*   **格式转换**: 统一转为 WAV 或 FLAC，去除 MP3 压缩伪影。
*   **VAD 清洗**: 使用 Voice Activity Detection 去除长静音片段，提高训练效率。
*   **分块 (Chunking)**: 将长音频切分为固定长度（如 30s），避免 OOM。

### 2.2 数据增强 (Augmentation)
为了提高模型鲁棒性，在训练时动态添加扰动。
*   **加噪**: 叠加环境音（雨声、街道声）。
*   **混响 (Reverb)**: 模拟不同房间的声学环境。
*   **变速变调**: 改变语速和音高。

---

## 3. 训练与微调 (Training & Fine-tuning)

### 3.1 显存优化
语音序列通常很长（1分钟音频对应 3000-4000 个 Token），显存压力大。
*   **Gradient Checkpointing**: 牺牲计算换显存。
*   **Flash Attention**: 必备组件，加速长序列注意力计算。

### 3.2 混合精度训练
*   **BF16**: 推荐使用 BFloat16，防止训练溢出。

---

## 4. 部署与监控 (Deployment & Monitoring)

### 4.1 核心指标
*   **RTF (Real Time Factor)**: 处理时长 / 音频时长。
    *   RTF < 1.0 表示比实时快。
    *   RTF > 1.0 表示比实时慢（不可接受）。
*   **首字延迟 (TTFT)**: 对于流式 ASR/TTS，用户从说话到看到第一个字（或听到第一个音）的时间。

### 4.2 质量监控
*   **静音检测**: 监控输出音频是否包含过长的静音。
*   **幻觉检测**: 监控 ASR 是否输出了音频中不存在的重复短语。
