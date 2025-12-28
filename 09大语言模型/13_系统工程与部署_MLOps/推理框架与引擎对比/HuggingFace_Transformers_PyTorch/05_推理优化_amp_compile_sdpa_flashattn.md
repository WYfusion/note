# 推理优化：AMP, Compile, SDPA 与 Flash Attention

PyTorch 2.x 和 Transformers 库提供了多种“免费”的加速手段，无需量化即可显著提升推理速度。

## 1. Flash Attention 2

Flash Attention (Dao et al.) 通过优化 GPU 显存读写（IO-aware），将 Attention 计算速度提升了数倍，且显存占用从 $O(N^2)$ 降为 $O(N)$。
对于 Audio LLM，由于音频序列通常很长（Whisper Encoder 序列长 1500/3000，Qwen-Audio 可达数万），Flash Attention 收益巨大。

### 1.1 启用方法
首先安装依赖：
```bash
pip install flash-attn --no-build-isolation
```

加载模型时指定 `attn_implementation="flash_attention_2"`：
```python
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    "openai/whisper-large-v3",
    torch_dtype=torch.float16,
    use_safetensors=True,
    attn_implementation="flash_attention_2" # 开启 FA2
).to("cuda")
```

### 1.2 限制
*   仅支持 Ampere (RTX 3090/A100) 或更新的 GPU 架构。
*   必须使用 FP16 或 BF16 精度（不支持 FP32）。

---

## 2. PyTorch SDPA (Scaled Dot Product Attention)

如果显卡不支持 Flash Attention 2（如 T4, V100），可以使用 PyTorch 原生的 SDPA 加速。它会自动选择最优的 Attention 实现（FlashAttention-1, Memory-Efficient Attention, 或 Math）。

### 2.1 启用方法
Transformers 默认会尝试使用 SDPA。
```python
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    "openai/whisper-large-v3",
    torch_dtype=torch.float16,
    attn_implementation="sdpa" # 默认值
).to("cuda")
```

---

## 3. torch.compile (PyTorch 2.0+)

`torch.compile` 使用 TorchInductor 编译器将 PyTorch 代码编译为优化的 Triton 内核。

### 3.1 针对 Audio Encoder 的优化
Whisper 的 Encoder 是纯 Transformer 结构，非常适合编译。

```python
import torch

# 加载模型
model = AutoModelForSpeechSeq2Seq.from_pretrained(...)
model.to("cuda")

# 编译模型
# mode="reduce-overhead": 适合小 Batch 推理，减少 Python 调用开销
# mode="max-autotune": 适合大 Batch 吞吐，编译时间长但运行最快
model.model.encoder = torch.compile(model.model.encoder, mode="reduce-overhead")

# 第一次推理会触发编译（较慢），之后会非常快
model.generate(...)
```

### 3.2 注意事项
*   **动态形状 (Dynamic Shapes)**: 如果输入音频长度变化剧烈，会导致频繁的重编译（Re-compilation）。建议配合 Padding 到固定长度使用，或设置 `dynamic=True`（但可能影响性能）。
*   **Graph Breaks**: 某些 Python 控制流会导致编译中断。Transformers 库的模型通常已经对编译友好。

---

## 4. 混合精度推理 (AMP)

始终使用 FP16 或 BF16。

*   **FP16**: 速度快，但数值范围小，可能溢出（Overflow）。
*   **BF16**: 数值范围与 FP32 相同，精度略低，训练和推理更稳定。需要 Ampere+ GPU。

```python
# 推荐配置
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

model = AutoModel.from_pretrained(..., torch_dtype=dtype)
```

## 5. 性能对比 (Whisper Large v3)

| 优化手段 | 相对速度 | 显存占用 | 备注 |
| :--- | :--- | :--- | :--- |
| Baseline (FP32) | 1.0x | 100% | |
| FP16 | 2.5x | 50% | 几乎无损 |
| FP16 + SDPA | 3.0x | 40% | 推荐默认 |
| FP16 + FA2 | 3.5x | 30% | 长序列优势明显 |
| FP16 + FA2 + Compile | 4.0x+ | 30% | 首次启动慢 |
