# 与 Serving 衔接：FastAPI 与 OpenAI 兼容接口

将 HuggingFace 模型封装为 HTTP 服务是落地的最后一步。FastAPI 是目前 Python 生态中最流行的微服务框架。

## 1. 基础架构

### 1.1 全局模型加载
模型加载是耗时操作，必须在服务启动时（Startup Event）完成，并作为全局变量或依赖注入。

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时加载
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "openai/whisper-large-v3"
    
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch.float16, use_safetensors=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    
    # 使用 pipeline 简化推理
    ml_models["whisper"] = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=device,
        torch_dtype=torch.float16
    )
    
    yield
    
    # 关闭时清理
    ml_models.clear()
    torch.cuda.empty_cache()

app = FastAPI(lifespan=lifespan)
```

---

## 2. 实现 OpenAI 兼容的 Audio API

OpenAI 的 `/v1/audio/transcriptions` 接口是事实标准。

### 2.1 接口定义
```python
from fastapi import File, UploadFile, Form, HTTPException
from typing import Optional
import shutil
import os

@app.post("/v1/audio/transcriptions")
async def transcribe_audio(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    language: Optional[str] = Form(None),
    response_format: str = Form("json")
):
    # 1. 保存上传的临时文件
    temp_filename = f"temp_{file.filename}"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # 2. 调用模型推理
        pipe = ml_models["whisper"]
        
        # Pipeline 支持直接传入文件路径，自动处理 ffmpeg 解码
        generate_kwargs = {"task": "transcribe"}
        if language:
            generate_kwargs["language"] = language
            
        result = pipe(
            temp_filename, 
            chunk_length_s=30, # 长音频切片
            batch_size=8,
            generate_kwargs=generate_kwargs
        )
        
        text = result["text"]
        
        # 3. 格式化输出
        if response_format == "json":
            return {"text": text}
        else:
            return text
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 4. 清理临时文件
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
```

---

## 3. 并发与性能 (Concurrency)

### 3.1 `async def` vs `def`
*   **`def`**: FastAPI 会在线程池中运行该函数。对于 CPU 密集型任务（如模型推理），这会阻塞线程池，但不会阻塞 Event Loop。
*   **`async def`**: 直接在 Event Loop 中运行。**千万不要在 `async def` 中直接运行耗时的 PyTorch 推理**，这会阻塞整个服务，导致无法响应心跳检查。

**最佳实践**:
1.  使用 `def` 定义推理接口，让 FastAPI 自动调度到线程池。
2.  或者在 `async def` 中使用 `await run_in_threadpool(inference_func)`。

### 3.2 批处理 (Batching)
上述代码是“来一个请求处理一个”。在高并发场景下，应该使用 **Dynamic Batching**。
*   可以使用 `mosec` 或 `ray serve` 等框架来实现请求队列和动态批处理。
*   简单实现：使用一个全局 `Queue`，后台启动一个 Consumer 线程不断从 Queue 取数据组成 Batch 进行推理。

---

## 4. 部署命令

使用 `uvicorn` 启动服务：

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```
*   **Workers**: 对于 GPU 推理服务，通常设置 `workers=1`，因为 GPU 是独占资源。多 Worker 会导致显存争抢和上下文切换开销。
