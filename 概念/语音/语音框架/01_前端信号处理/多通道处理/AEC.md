# 回声消除 (Acoustic Echo Cancellation, AEC)

## 作用
消除扬声器播放的声音被麦克风重新采集形成的回声，避免"自己听到自己"的问题。

## 应用场景
- 智能音箱（边播放音乐边语音交互）
- 视频会议
- 车载语音系统

## 核心方法
- 自适应滤波器（LMS、NLMS、RLS）
- 非线性回声消除（NLMS + 后处理）
- 深度学习方法（DTLN、FullSubNet）
