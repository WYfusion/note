多模态嵌入将文本、图像、音频、视频等不同模态的数据映射到**同一个共享语义空间**，实现跨模态检索。

<aside>
💡

**通俗理解**：一个"通用翻译器"，不管输入是文字、图片还是声音，都翻译成同一种"语义语言"，让不同模态之间可以直接比较。

</aside>

---

## 核心概念

- **共享语义空间**：所有模态的向量在同一空间中，语义相近则距离近
- **代表方向**：CLIP（图文）、CLAP（音文）、ImageBind（6 模态）、Jina Embeddings v4、Nomic Multimodal
- **用途**：跨模态检索、统一知识库、富媒体 RAG、agent 工具检索

---

## 趋势

- **单模型统一多模态**：一个模型处理所有输入类型
- **同时支持 dense + multi-vector**：兼顾效率和精度
- **支持复杂文档**：长文档 / PDF / 图表 / OCR-rich 页面

---

## Python 实战：ImageBind 多模态嵌入

```python
# ImageBind 支持 6 种模态：图像、文本、音频、深度、热图、IMU
# 以下为概念性示例（需安装 imagebind 包）

import torch
from imagebind import data as imagebind_data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

# 加载模型
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()

# 准备不同模态的输入
inputs = {
    ModalityType.TEXT: imagebind_data.load_and_transform_text(
        ["A dog playing in the park", "Ocean waves"], device="cpu"
    ),
    ModalityType.VISION: imagebind_data.load_and_transform_vision_data(
        ["dog.jpg", "ocean.jpg"], device="cpu"
    ),
    ModalityType.AUDIO: imagebind_data.load_and_transform_audio_data(
        ["dog_bark.wav", "waves.wav"], device="cpu"
    ),
}

# 编码所有模态
with torch.no_grad():
    embeddings = model(inputs)

# 跨模态相似度：文本 vs 图像
text_embeds = embeddings[ModalityType.TEXT]
image_embeds = embeddings[ModalityType.VISION]
similarity = torch.softmax(
    text_embeds @ image_embeds.T, dim=-1
)
print("文本 → 图像 相似度:")
print(similarity)

# 跨模态相似度：文本 vs 音频
audio_embeds = embeddings[ModalityType.AUDIO]
similarity_ta = torch.softmax(
    text_embeds @ audio_embeds.T, dim=-1
)
print("文本 → 音频 相似度:")
print(similarity_ta)
```

---

## 多模态统一检索架构

```python
# 统一多模态知识库检索伪代码
class MultimodalRetriever:
    def __init__(self, model, vector_db):
        self.model = model
        self.db = vector_db  # 如 Qdrant / Milvus
    
    def index(self, items):
        """索引不同模态的内容"""
        for item in items:
            if item.type == "text":
                vec = self.model.encode_text(item.content)
            elif item.type == "image":
                vec = self.model.encode_image(item.content)
            elif item.type == "audio":
                vec = self.model.encode_audio(item.content)
            self.db.upsert(id=item.id, vector=vec, metadata={"type": item.type})
    
    def search(self, query, query_type="text", top_k=10):
        """跨模态检索：任意模态 query → 任意模态 results"""
        if query_type == "text":
            q_vec = self.model.encode_text(query)
        elif query_type == "image":
            q_vec = self.model.encode_image(query)
        return self.db.search(vector=q_vec, limit=top_k)
```