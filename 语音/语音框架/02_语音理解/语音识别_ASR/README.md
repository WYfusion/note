# 语音识别 (Automatic Speech Recognition, ASR)

## 作用
将语音信号转换为文本，是语音系统的核心任务。

## 主流架构
- **CTC**: 无需对齐，适合流式
- **Attention (Seq2Seq)**: 效果好，非流式
- **RNN-T (Transducer)**: 流式+高精度
- **Hybrid CTC-Attention**: 结合两者优势

## SOTA模型
- **Whisper**: OpenAI多语言大模型
- **Conformer**: Convolution + Transformer

## 关键技术
- 语言模型融合 (LM Fusion)
- 流式/非流式推理
- 热词增强 (Contextual Biasing)
