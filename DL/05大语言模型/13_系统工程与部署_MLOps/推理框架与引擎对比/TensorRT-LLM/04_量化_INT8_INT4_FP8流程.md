# 量化流程：INT8, INT4 与 FP8

TensorRT-LLM 提供了业界最强的量化支持，能够显著降低显存占用并提升吞吐量。对于参数量巨大的 Audio LLM（如 Qwen-Audio-Chat 7B/14B），量化往往是部署的必选项。

## 1. 量化类型概览

| 模式 | 权重精度 | 激活精度 | 适用场景 | 硬件要求 |
| :--- | :--- | :--- | :--- | :--- |
| **W8A16** | INT8 | FP16 | 显存受限，计算不受限 | 所有 GPU |
| **W4A16 (AWQ/GPTQ)** | INT4 | FP16 | 极致显存压缩 (7B 模型仅需 ~4GB) | Ampere (3090/A10) 及以上 |
| **SmoothQuant (INT8)** | INT8 | INT8 | 极致吞吐，全链路 INT8 计算 | Ampere 及以上 |
| **FP8** | FP8 | FP8 | H100/L40S 专属，兼顾精度与速度 | Hopper 架构 |

## 2. 量化工具：ModelOpt (原 AMMO)

NVIDIA 提供了 `nvidia-modelopt` 库（之前叫 AMMO），用于对 HF 模型进行校准（Calibration）和导出。

### 2.1 安装
```bash
pip install nvidia-modelopt[torch]
```

### 2.2 AWQ 量化示例 (INT4)
AWQ (Activation-aware Weight Quantization) 是目前最流行的权重量化方法，对精度损失极小。

```python
import modelopt.torch.quantization as mtq

# 1. 加载 HF 模型
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-Audio-Chat")

# 2. 定义量化配置 (INT4 AWQ)
quant_cfg = mtq.INT4_AWQ_CFG

# 3. 校准 (Calibration) - 需要少量校准数据
# 对于 Audio 模型，这里需要喂入真实的 Audio Feature + Text
def calibrate_loop(model):
    for batch in calibration_dataloader:
        model(batch)

# 4. 执行量化
mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)

# 5. 导出为 TensorRT-LLM 格式 checkpoint
mtq.save(model, "./tllm_checkpoint_int4")
```

## 3. 构建量化引擎

获得量化后的 Checkpoint 后，在 `trtllm-build` 阶段需要指定对应的标志。

```bash
trtllm-build \
    --checkpoint_dir ./tllm_checkpoint_int4 \
    --output_dir ./tllm_engines_int4 \
    --use_weight_only \
    --weight_only_precision int4_awq
```

## 4. 语音模型的特殊考虑

### 4.1 Audio Encoder 量化
*   **慎重量化 Encoder**: Audio Encoder (如 Whisper Encoder) 参数量通常较小（~600M），但对精度非常敏感。
*   **建议**: Encoder 保持 FP16，仅对 LLM Decoder 进行 INT4/INT8 量化。
*   **混合精度**: TensorRT-LLM 允许 Encoder 和 Decoder 使用不同的精度。

### 4.2 精度损失评估
语音任务（ASR）对 WER (Word Error Rate) 敏感。
*   INT4 量化通常会导致 WER 轻微上升（0.5% - 1%）。
*   FP8 量化（如果硬件支持）通常能提供比 INT8 更好的精度保持。

## 5. 性能收益
*   **显存**: INT4 相比 FP16 节省约 3 倍显存。
*   **带宽**: 减少了从显存读取权重的带宽压力，对于 Memory-Bound 的解码阶段，速度提升明显。
