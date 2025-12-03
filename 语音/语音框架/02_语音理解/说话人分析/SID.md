# 说话人识别 (Speaker Identification, SID)

## 作用
判断"这段语音是谁说的"，从已知说话人集合中识别身份。

## 与声纹验证的区别
- SID: 1:N 识别（多分类）
- SV: 1:1 验证（二分类）

## 应用场景
- 会议记录（标注发言人）
- 个性化服务
- 安防监控

## 核心方法
- x-vector / ECAPA-TDNN
- ResNet-based Speaker Embedding
- 基于SSL的说话人表征
