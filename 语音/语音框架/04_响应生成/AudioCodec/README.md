# 音频编解码 (Audio Codec)

## 作用
将连续音频信号压缩为离散token表示，是Speech LM的基础。

## 代表模型

### EnCodec (Meta)
- RVQ量化
- 多码率支持

### SoundStream (Google)
- 类似架构
- 用于AudioLM

### DAC (Descript Audio Codec)
- 高质量音频压缩

## 核心技术

### RVQ (残差向量量化)
多层级量化，逐层细化

### 语义Token
- 来自SSL模型（如HuBERT）
- 捕获语义信息
- 与声学Token配合使用
