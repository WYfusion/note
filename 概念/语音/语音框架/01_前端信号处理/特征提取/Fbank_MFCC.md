# 声学特征提取 (Fbank / MFCC)

## 作用
将时域波形转换为频域特征表示，供后续模型使用。

## Fbank (Filter Bank)
- 对频谱应用Mel滤波器组
- 保留更多信息，现代ASR主流特征
- 通常40-80维

## MFCC (Mel-Frequency Cepstral Coefficients)
- 在Fbank基础上做DCT变换
- 去相关，传统ASR常用
- 通常13维 + delta + delta-delta

## 其他特征
- Pitch/F0：基频特征
- SSL特征：Wav2Vec/HuBERT输出
