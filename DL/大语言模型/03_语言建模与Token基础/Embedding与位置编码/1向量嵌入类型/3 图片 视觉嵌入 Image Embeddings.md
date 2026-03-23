图片嵌入将图像映射为向量，支持图像检索、图文匹配和视觉文档理解。

<aside>
💡

**通俗理解**：给每张图片生成一个"视觉指纹"。可以用文字搜图片，也可以用图片搜相似图片。

</aside>

---

## 三大方向

### 单模态视觉嵌入

- **代表**：ViT、DINOv2
- **原理**：纯视觉编码器，将图像编码为向量
- **用途**：图像聚类、近重复检索、视觉相似图搜索

### 图文共享嵌入

- **代表**：CLIP、SigLIP、SigLIP2、Jina CLIP
- **原理**：图像和文本编码到**同一向量空间**
- **用途**：text→image、image→text、image→image 检索；开放词汇分类

### 文档/PDF/图表视觉嵌入

- **代表**：Nomic Embed Multimodal、Jina Embeddings v4
- **原理**：将视觉丰富的文档页面直接编码，无需 OCR
- **用途**：图表、扫描页、视觉丰富 PDF、论文页级检索

---

## Python 实战：CLIP 图文检索

```python
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# 加载 CLIP
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 准备图片和文本
images = [Image.open(f"img_{i}.jpg") for i in range(5)]  # 5 张图
texts = ["一只在草地上奔跑的狗", "城市夜景", "海边日落"]

# 编码图片
img_inputs = processor(images=images, return_tensors="pt", padding=True)
with torch.no_grad():
    img_embeds = model.get_image_features(**img_inputs)
    img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)

# 编码文本
txt_inputs = processor(text=texts, return_tensors="pt", padding=True)
with torch.no_grad():
    txt_embeds = model.get_text_features(**txt_inputs)
    txt_embeds = txt_embeds / txt_embeds.norm(dim=-1, keepdim=True)

# text → image 检索：计算相似度矩阵
similarity = txt_embeds @ img_embeds.T  # [n_texts, n_images]
print("文本→图片相似度矩阵:")
print(similarity)

# 对每个文本，找最相似的图片
for i, text in enumerate(texts):
    best_img_idx = similarity[i].argmax().item()
    print(f"'{text}' → 最匹配: img_{best_img_idx}.jpg (score={similarity[i][best_img_idx]:.3f})")
```

---

## DINOv2 单模态视觉嵌入

```python
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image

processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
model = AutoModel.from_pretrained("facebook/dinov2-base")

def get_image_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    # 使用 [CLS] token 作为图像表示
    return outputs.last_hidden_state[:, 0, :]

# 计算两张图片的相似度
emb1 = get_image_embedding("cat1.jpg")
emb2 = get_image_embedding("cat2.jpg")
cosine_sim = torch.nn.functional.cosine_similarity(emb1, emb2)
print(f"图片相似度: {cosine_sim.item():.4f}")
```