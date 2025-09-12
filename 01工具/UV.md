## Ubuntu安装UV
稳健的增加超时时间重试
```bash
curl -LsSf --connect-timeout 60 --max-time 300 https://astral.sh/uv/install.sh | sh
```

## 查看UV版本
```bash
uv --version
```


## 安装python
```bash
# 查看可用python版本
uv python list
# 下载合适版本
uv python install cpython-3.12
```


## 创建venv环境
```bash
# uv init -p [版本号] [环境名] 如若版本号未下载，则会自动下载，若已经下载，则直接使用本地已经缓存的；环境名可以不设置，默认为venv
uv init -p 3.12
# 该命令会自动初始化符合规范的python项目基本文件于当前路径下，例如pyproject.toml、.gitignore、.python-version、README.md

# 推荐使用add命令添加依赖，add命令可以自动的将依赖同步到pyproject.toml中去，推荐先配置主要依赖，例如torch，添加依赖后会自动在当前路径下面创建.venv虚拟环境
uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
查看GPU支持的最高版本
```bash
nvidia-smi # 查看 CUDA 版本
```

Pytorch安装可以看：[Previous PyTorch Versions](https://pytorch.org/get-started/previous-versions/)


## 激活venv环境

### Mac、Linux:
```bash
# source [环境路径]
source /home/gzhu/data1/wy/Data_processing/.venv/bin/activate
```

### Windows:
Windows PowerShell 有严格的执行策略（默认是 `Restricted`），防止未授权的脚本运行。虚拟环境的 `activate.ps1` 是一个 PowerShell 脚本，因此被默认阻止。
临时方法：临时允许运行脚本（仅对当前窗口有效，关闭后恢复默认设置）
```powershell
Set-ExecutionPolicy Bypass -Scope Process -Force
```
这个命令的作用是：在当前进程（PowerShell 窗口）中，临时将执行策略改为 `Bypass`（绕过限制），不会影响系统全局设置，安全性较高。
```powershell
.\venv\Scripts\activate
```

如果要永久修改执行策略(不推荐，保持系统默认的安全策略)
```powershell
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## 查看venv环境中依赖关系树
```bash
uv tree
```

## 自动配置
示例：
```toml
[project]
name = "vad"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "numpy>=1.21.0",
    "torch>=2.0.0,<2.8.0",
    "torchvision", 
    "torchaudio",
]

[[tool.uv.index]]
name = "pytorch"
url = "https://download.pytorch.org/whl/cu118"
explicit = true

```
通过已经编辑好的pyproject.toml文件自动配置uv环境依赖
```bash
uv sync
```