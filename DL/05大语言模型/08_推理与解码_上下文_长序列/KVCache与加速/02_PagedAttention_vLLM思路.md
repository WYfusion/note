# PagedAttention 与 vLLM：显存管理革命

## 1. 显存碎片化问题
在传统的 KV Cache 管理中，显存通常是预分配的（预设最大长度）。
- **内部碎片**：如果预设长度 2048，实际只用了 100，浪费 95%。
- **外部碎片**：不同请求的 KV Cache 存储在不连续的物理内存中，难以合并。

## 2. PagedAttention 原理
受操作系统虚拟内存（Virtual Memory）分页机制的启发，vLLM 提出了 PagedAttention。

### 2.1 核心机制
- **Block**：将 KV Cache 切分为固定大小的块（Block），例如每块包含 16 个 Token。
- **Block Table**：维护一个逻辑块到物理块的映射表。
- **非连续存储**：逻辑上连续的 Token，其 KV Cache 在物理显存中可以是不连续的。

### 2.2 优势
- **零浪费**：按需分配，没有预留空间，内部碎片极小（最后一个 Block 未填满）。
- **共享内存**：在 Beam Search 或 Parallel Sampling 中，不同的序列可以共享公共前缀（Prefix）的物理块，大幅节省显存。

## 3. 语音大模型中的应用 (Audio vLLM)

### 3.1 变长请求的极致适配
语音对话的长度差异极大（从 1秒的"嗯"到 1小时的演讲）。
- **传统方式**：必须按最长音频 Padding，浪费严重。
- **PagedAttention**：完美处理变长音频流，无需 Padding，显存利用率接近 100%。

### 3.2 并发服务 (Serving)
在语音助手场景中，高并发是常态。
- **Throughput 提升**：由于显存节省，Audio LLM 可以支持更大的 Batch Size。
- **流式传输 (Streaming)**：PagedAttention 天然支持流式生成，每生成一个 Block 就可以发送音频数据，降低首字延迟（Time-to-First-Audio）。

### 3.3 跨模态共享
在图文/语音多模态模型中，System Prompt（如"你是一个语音助手..."）通常是固定的。
- **Prefix Sharing**：所有用户的请求都可以共享这一段 System Prompt 的 KV Cache Block，对于包含长 System Prompt 的 Audio LLM 尤为有效。
