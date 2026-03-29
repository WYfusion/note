# 工程化：权重格式与 Safetensors

在模型部署工程中，权重文件的加载速度和安全性往往被忽视，但它们直接影响服务的冷启动时间（Cold Start Time）和安全性。

## 1. 传统格式：PyTorch Bin (Pickle)

传统的 `.bin` 或 `.pth` 文件使用 Python 的 `pickle` 模块进行序列化。
*   **安全风险**: `pickle` 可以执行任意代码。加载不信任的 `.bin` 文件可能导致服务器被入侵。
*   **加载效率**: 需要先将文件读入 CPU 内存，反序列化，然后再拷贝到 GPU 显存。存在多余的内存拷贝。

## 2. 现代标准：Safetensors

HuggingFace 推出的 `safetensors` 格式已成为大模型存储的事实标准。

### 2.1 核心优势
1.  **安全性 (Safety)**: 纯数据格式，不包含代码，杜绝了 Pickle 反序列化攻击。
2.  **零拷贝加载 (Zero-copy / Memory Mapping)**:
    *   利用操作系统的 `mmap` 机制，直接将磁盘文件映射到内存地址空间。
    *   从 CPU 内存到 GPU 显存的传输更高效。
    *   **显著降低冷启动时间**：对于 10GB+ 的模型，加载速度可提升数倍。
3.  **懒加载 (Lazy Loading)**: 可以只加载模型的一部分权重（例如只加载 Encoder），而无需读取整个文件。

### 2.2 在 Transformers 中使用
现代的 Transformers 版本默认优先加载 `model.safetensors`。

```python
# 显式指定
model = AutoModel.from_pretrained(
    "openai/whisper-large-v3",
    use_safetensors=True
)
```

---

## 3. 权重转换实战

如果你手中的模型只有 `.bin` 文件，建议将其转换为 `.safetensors`。

### 3.1 使用脚本转换
HuggingFace 提供了转换脚本：

```python
import torch
from safetensors.torch import save_file
from transformers import AutoModel

model_id = "path/to/old_model"
model = AutoModel.from_pretrained(model_id)

# 获取 state_dict
state_dict = model.state_dict()

# 保存为 safetensors
save_file(state_dict, "model.safetensors")
```

### 3.2 模型切分 (Sharding)
对于超大模型（如 Qwen-Audio-Chat 7B），单个文件可能超过 10GB，不便传输。通常将其切分为多个 2GB-5GB 的分片（Shard）。

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-Audio-Chat")

# 保存并自动切分
model.save_pretrained(
    "local_dir", 
    max_shard_size="2GB", 
    safe_serialization=True # 保存为 safetensors
)
```
输出目录结构：
```
local_dir/
  config.json
  model.safetensors.index.json  # 索引文件，记录每个权重在哪个分片
  model-00001-of-00004.safetensors
  model-00002-of-00004.safetensors
  ...
```

---

## 4. 最佳实践

1.  **生产环境强制 Safetensors**: 在加载模型时设置 `use_safetensors=True`，如果只有 `.bin` 则报错，确保安全。
2.  **利用 SSD**: `mmap` 的性能高度依赖磁盘 I/O，务必将模型存放在 NVMe SSD 上。
3.  **多进程共享**: 在 Linux 上，多个进程加载同一个 `safetensors` 文件时，操作系统会共享 Page Cache，从而节省物理内存（对于多 Worker 的推理服务非常有用）。
