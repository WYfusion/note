# 语音活动检测 (Voice Activity Detection, VAD)

## 作用
判断音频帧中是否包含语音，区分语音段和静音/噪声段。

## 应用场景
- ASR前端（只处理有语音的部分）
- 通信系统（静音压缩）
- 语音分段

## 核心方法
- 能量/过零率检测
- GMM-based VAD
- DNN-based VAD (WebRTC VAD, Silero VAD)
