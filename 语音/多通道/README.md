# 多通道音频信号处理知识体系

本目录包含多通道音频信号处理的完整知识体系，涵盖理论基础、经典算法和深度学习方法。

---

## 📂 目录结构

### 1. 入门文档
- [[多通道音频信号处理入门]] - 总览、信号模型、任务分类

### 2. 盲源分离 (BSS)
📁 **BSS/** - 11个详细文档
- 基础理论：数学基础、ICA理论
- 经典算法：FastICA、JADE、IVA、AuxIVA、ILRMA
- 深度学习：TasNet、Conv-TasNet、DPRNN、SepFormer

### 3. 波束形成 (Beamforming)
📁 **Beamforming/** - 已创建3个文档
- 00_波束形成概述
- 01_波束形成数学基础
- 06_MVDR波束形成

**待补充**：
- 延迟求和、超指向性、差分阵列
- GSC、LCMV
- 深度学习波束形成

### 4. 声源定位 (Source Localization)
📁 **SourceLocalization/** - 已创建1个文档
- 00_声源定位概述

**待补充**：
- GCC-PHAT、SRP-PHAT
- MUSIC、ESPRIT
- 深度学习定位

### 5. 去混响 (Dereverberation)
📁 **Dereverberation/** - 已创建1个文档
- 00_去混响概述

**待补充**：
- 混响模型、WPE算法
- 逆滤波、谱减法
- 深度学习去混响

### 6. 语音增强 (Speech Enhancement)
📁 **SpeechEnhancement/** - 已创建1个文档
- 00_语音增强概述

**待补充**：
- 维纳滤波、谱减法
- MWF多通道维纳滤波
- 深度学习增强

### 7. 说话人追踪 (Speaker Tracking)
📁 **SpeakerTracking/** - 已创建1个文档
- 00_说话人追踪概述

**待补充**：
- 卡尔曼滤波、粒子滤波
- 多假设追踪
- 深度学习追踪

---

## 🎯 学习路径建议

### 初学者路径
1. 阅读[[多通道音频信号处理入门]]
2. 学习[[BSS/01_BSS数学基础]]
3. 实践[[Beamforming/01_波束形成数学基础]]
4. 了解[[SourceLocalization/00_声源定位概述]]

### 进阶路径
1. 深入学习[[BSS/02_ICA独立成分分析]]
2. 掌握[[Beamforming/06_MVDR波束形成]]
3. 研究经典算法的数学推导
4. 实现基本算法

### 高级路径
1. 学习深度学习方法
2. 研究最新论文
3. 解决实际工程问题
4. 开发创新算法

---

## 📊 已完成内容统计

| 任务 | 概述 | 基础理论 | 经典算法 | 深度学习 | 总计 |
|------|------|----------|----------|----------|------|
| BSS | ✅ | ✅ | ✅✅✅✅✅ | ✅✅✅✅ | 11 |
| Beamforming | ✅ | ✅ | ✅ | ⬜ | 3 |
| SourceLoc | ✅ | ⬜ | ⬜ | ⬜ | 1 |
| Dereverb | ✅ | ⬜ | ⬜ | ⬜ | 1 |
| Enhancement | ✅ | ⬜ | ⬜ | ⬜ | 1 |
| Tracking | ✅ | ⬜ | ⬜ | ⬜ | 1 |
| **总计** | 6 | 2 | 6 | 4 | **18** |

---

## 🔧 实用工具

### Python库
```python
# 房间声学仿真
import pyroomacoustics as pra

# 音频处理
import librosa
import soundfile as sf

# 深度学习
import torch
from asteroid import models
```

### 开源项目
- **pyroomacoustics**: 房间声学和波束形成
- **asteroid**: 音频源分离
- **ESPnet**: 端到端语音处理
- **ODAS**: 开源定位和追踪

### 数据集
- **WSJ0-2mix**: 语音分离
- **CHiME**: 多通道识别
- **REVERB**: 混响语音
- **LOCATA**: 定位和追踪

---

## 📚 参考资源

### 经典教材
1. *Microphone Array Signal Processing* - Benesty et al.
2. *Acoustic Array Systems* - Brandstein & Ward
3. *Speech Enhancement* - Loizou
4. *Fundamentals of Signal Processing* - Oppenheim

### 重要会议
- ICASSP (IEEE)
- INTERSPEECH
- WASPAA
- EUSIPCO

### 期刊
- IEEE/ACM Transactions on Audio, Speech, and Language Processing
- Speech Communication
- EURASIP Journal on Audio, Speech, and Music Processing

---

## 🚀 后续计划

### 短期目标
1. 补充波束形成的经典算法（延迟求和、GSC）
2. 完善声源定位的详细推导（GCC-PHAT、SRP-PHAT）
3. 添加去混响的WPE算法详解

### 中期目标
1. 补充所有任务的深度学习方法
2. 添加实际代码示例
3. 创建综合案例研究

### 长期目标
1. 建立完整的实验框架
2. 提供端到端的项目示例
3. 持续更新最新研究进展

---

## 💡 贡献指南

本知识体系持续更新中，欢迎补充和完善：
- 修正错误和不准确的描述
- 补充缺失的算法和方法
- 添加实际应用案例
- 提供代码实现示例

---

## 📝 更新日志

**2024-11-25**
- ✅ 创建多通道音频处理入门文档
- ✅ 完成BSS完整知识体系（11个文档）
- ✅ 创建波束形成基础文档（3个）
- ✅ 创建其他任务概述文档（4个）
- 📊 总计18个详细文档

**待更新**
- 补充波束形成经典算法
- 补充声源定位详细推导
- 补充去混响和语音增强算法
- 添加说话人追踪详细内容
