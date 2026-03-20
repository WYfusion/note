# 语音合成 (Text-to-Speech, TTS)

## 作用
将文本转换为自然语音。

## 流程

### 文本前端
- G2P (字音转换)
- 韵律预测
- 多音字消歧
- BERT韵律建模

### 声学模型
- Tacotron: 端到端Attention
- FastSpeech: 非自回归，快速
- VITS: 端到端变分推断
- VALL-E: 语言模型范式
- CosyVoice: 最新SOTA

### 声码器 (Vocoder)
- WaveNet: 自回归，高质量
- HiFi-GAN: GAN-based，快速
- BigVGAN: 大规模HiFi-GAN
- Vocos: 轻量级
- Neural Codec: 离散token解码
