# Volume 与 Bind Mount

## 适用场景
- **代码调试**：在宿主机写代码，容器里实时运行（Bind Mount）。
- **数据持久化**：删除容器后，数据库文件不丢失（Volume）。
- **数据集共享**：多个容器共用一份 500GB 的 ImageNet 数据集。

## 1. 核心对比：Volume vs Bind Mount
Docker 提供了两种主要的数据挂载方式，**深度学习场景下 90% 的情况推荐使用 Bind Mount**。

| 特性 | Volume (数据卷) | Bind Mount (绑定挂载) |
| :--- | :--- | :--- |
| **存储位置** | Docker 内部管理 (`/var/lib/docker/volumes/...`) | **宿主机任意路径** (如 `D:\Code`, `/home/user/data`) |
| **管理方式** | `docker volume create/ls/rm` | 用户手动管理文件夹 |
| **适用场景** | 数据库 (MySQL/Redis)、不想关心存储位置的数据 | **源代码、数据集、模型权重、配置文件** |
| **优点** | 跨平台一致性好，性能略高 (Linux下) | **直观**，方便在宿主机用 VS Code 编辑代码 |

## 2. Bind Mount (推荐)
直接把宿主机的目录“映射”到容器里。

### 语法
```bash
-v <宿主机绝对路径>:<容器内路径>[:ro]
```

### 深度学习常用示例
```bash
docker run -d \
  --gpus all \
  # 1. 挂载代码 (方便修改)
  -v /home/user/project:/workspace/project \
  # 2. 挂载数据集 (只读，防止误删)
  -v /data/imagenet:/data/imagenet:ro \
  # 3. 挂载模型保存目录
  -v /home/user/checkpoints:/workspace/checkpoints \
  pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
```

> [!TIP] 实时生效
> Bind Mount 是实时的。你在宿主机修改了 `train.py`，容器里立即就能看到修改后的文件，无需重启容器。

## 3. Volume (数据卷)
由 Docker 自动管理，适合持久化那些“不需要人工干预”的数据。

### 常用操作
```bash
# 1. 创建卷
docker volume create my-db-data

# 2. 使用卷 (注意：冒号左边是卷名，不是路径)
docker run -d -v my-db-data:/var/lib/mysql mysql

# 3. 查看卷信息 (找它到底存在哪)
docker volume inspect my-db-data
# 输出 Mountpoint: /var/lib/docker/volumes/my-db-data/_data

# 4. 清理未使用的卷
docker volume prune
```

## 4. 权限问题 (Permission Denied)
**这是 Bind Mount 最常见的坑。**
- **现象**：容器里 `pip install` 或 `mkdir` 报错 `Permission denied`。
- **原因**：宿主机目录属于 `root` 或其他用户，而容器内默认是 `root` (uid=0) 或者特定用户 (uid=1000)。如果 UID 不匹配，Linux 内核会拒绝写入。
- **解决**：
  1. **粗暴法**：宿主机 `chmod 777 -R /my/data`。
  2. **优雅法**：启动时指定用户 ID。
     ```bash
     docker run -u $(id -u):$(id -g) -v ...
     ```

## 总结
- **写代码/跑实验**：用 **Bind Mount** (`-v /host/path:/container/path`)。
- **跑数据库**：用 **Volume** (`-v vol_name:/data`)。
- **保护数据集**：加上 `:ro` (只读)。

