# 自监督预训练 (Self-Supervised Learning, SSL)

## 作用
从大量无标注语音数据中学习通用语音表征，可迁移到各种下游任务。

## 代表模型

### Wav2Vec 2.0 (Meta)
对比学习 + 量化

### HuBERT (Meta)
聚类伪标签 + 掩码预测

### WavLM (Microsoft)
加入去噪目标，多任务SOTA

### Data2Vec
统一的自监督框架

### BEST-RQ
随机投影量化

### Whisper Encoder
大规模弱监督预训练

## 特征层级
- 浅层：声学/说话人信息
- 中层：音素/语言信息
- 深层：语义信息

## 下游任务
ASR、SID、SER、LID、SLU等几乎所有语音任务
