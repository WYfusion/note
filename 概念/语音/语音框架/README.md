# 现代语音系统框架

本目录按照现代语音系统架构组织各个子任务模块。

## 目录结构

```
语音框架/
├── 01_前端信号处理/     # Front-End DSP
├── 02_语音理解/         # Speech Understanding
├── 03_对话与知识/       # Dialog & Knowledge
├── 04_响应生成/         # Response Generation
├── 05_端到端模型/       # E2E Models
└── 06_自监督预训练/     # SSL
```

## 主数据流

```
用户 → 麦克风 → 前端DSP → ASR → NLU → DM → NLG → TTS → 扬声器 → 用户
```

## 参考架构图

详见 `语音系统架构.drawio`
