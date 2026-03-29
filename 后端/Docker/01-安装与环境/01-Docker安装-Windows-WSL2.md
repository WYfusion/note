# Docker 安装（Windows + WSL2，推荐方案）

> 核心结论：**Windows 上最省心的 Docker 体验通常是 Docker Desktop + WSL2 后端**。

## 适用场景
- 日常开发（Python/Node/Java/GPU 深度学习）。
- 希望在 Windows 上获得接近 Linux 的容器体验。
- 需要良好的文件挂载、网络、IDE 集成。

## 你将得到什么
- 在 Windows 上安装并启用 Docker Desktop（使用 WSL2 backend）。
- 选择并配置 WSL2 发行版（Ubuntu 等），让它能执行 `docker`。
- 学会验证安装与常用排障入口。
- 掌握 PowerShell 下常用命令参数（`-p` / `-v` / `-e` / `--name` / `--gpus` 等）。

## 安装前检查（建议先做）
### 1) 系统要求
- Windows 10/11（建议较新版本）。
- CPU 支持虚拟化，并在 BIOS 中开启（Intel VT-x / AMD-V）。

### 2) WSL2 状态检查
在 PowerShell 里查看 WSL 是否可用：

```powershell
wsl --status
wsl -l -v
```

常见判断：
- `VERSION` 为 2 表示 WSL2。
- 如果没有发行版，可以先安装 Ubuntu。

## 安装步骤（推荐顺序）
### 步骤 A：安装 WSL2（如果尚未安装）
一般可以直接安装默认发行版：

```powershell
wsl --install
```

如果系统已装 WSL，但默认不是 WSL2：

```powershell
wsl --set-default-version 2
```

### 步骤 B：安装 Docker Desktop
1. 从 Docker Desktop 官方下载安装包并安装。
2. 安装完成后启动 Docker Desktop。
3. Docker Desktop 设置中确保：
	 - **Use the WSL 2 based engine** 已启用（通常默认）。
	 - 在 **Resources → WSL Integration** 中勾选你要集成的发行版（比如 Ubuntu）。

> 注意：如果你启用了 Hyper-V/WSL2 多种后端，通常仍建议以 WSL2 为主。

## 安装完成后的验证（很重要）
### 1) 在 PowerShell 验证

```powershell
docker version
docker info
```

重点关注：
- `Server:` 是否能显示（只显示 Client 往往意味着 Docker Desktop 没启动或引擎不可用）。
- `Docker Root Dir`（磁盘占用排查很有用）。

### 2) 跑一个 hello-world

```powershell
docker run --rm hello-world
```

如果能看到 "Hello from Docker!" 说明整个链路通了（拉取镜像 + 运行容器）。

## Docker Desktop 常用配置
### 1) 资源分配（CPU/内存/磁盘）
入口：Docker Desktop → Settings → Resources。

建议：
- 开发环境不要把内存拉太满（给 Windows 留余量）。
- 磁盘占用持续增长时，优先用 `docker system df` 查看，再决定是否 prune。

### 2) WSL Integration（决定你的 Linux 发行版里能不能用 docker）
入口：Settings → Resources → WSL Integration。

现象与原因：
- **在 WSL 里输入 `docker` 报命令不存在**：通常是没勾选该发行版集成，或发行版没装 docker-cli。

### 3) 文件共享与挂载性能
常见做法：
- **代码放在 WSL2 文件系统里**（例如 `\\wsl$\Ubuntu\home\...` 对应的 Linux 路径），再在 WSL 中运行 `docker` 构建/挂载，通常性能与稳定性更好。
- 直接挂载 Windows 盘符目录（如 `D:\project`）也能用，但遇到权限/换行/大小写等问题的概率更高。

## 常用命令与关键参数示例（PowerShell 版）
> 这里的示例更偏“你日常会用到”的参数组合。

### 1) 端口映射 `-p`
把容器的 80 映射到宿主机 8080：

```powershell
docker run --rm -p 8080:80 nginx
```

只允许本机访问（更安全，局域网其它机器访问不了）：

```powershell
docker run --rm -p 127.0.0.1:8080:80 nginx
```

### 2) 目录挂载 `-v`（PowerShell 用 `${PWD}`）
把当前目录挂载到容器 `/workspace`：

```powershell
docker run --rm -it -v ${PWD}:/workspace ubuntu bash
```

常见注意事项：
- PowerShell 用 **`${PWD}`**；CMD 用 `%cd%`；Git Bash 用 `$(pwd)`。
- 路径包含空格时要注意引号：`-v "${PWD}:/workspace"`（根据实际情况）。

### 3) 环境变量 `-e`

```powershell
docker run --rm -e "APP_ENV=dev" -e "HTTP_PROXY=http://proxy:8080" ubuntu env
```

### 4) 容器命名 `--name`

```powershell
docker run -d --name web -p 8080:80 nginx
docker ps
docker rm -f web
```

### 5) GPU（需要 NVIDIA 驱动 + Docker Desktop GPU 支持）
查看 GPU 是否能在容器里识别（示例镜像仅为说明思路）：

```powershell
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

常见参数：
- `--gpus all`：使用全部 GPU
- `--gpus "device=0"`：只使用 0 号 GPU

### 6) 共享内存 `--shm-size`
深度学习/多进程 DataLoader 常用：

```powershell
docker run --rm --shm-size=8g -it ubuntu bash
```

## 网络与代理的常见坑（Windows/WSL2 特别版）
- 能上网但 `docker pull` 很慢/超时：优先看 `04-镜像加速与代理.md`。
- 公司代理常见需要同时配置：
	- Docker Desktop 的代理
	- WSL2 内部的代理（如果你在 WSL2 中直接访问网络）
- `x509: certificate signed by unknown authority`：通常是自签证书/企业 MITM，必须把 CA 配进 Docker。

## 快速排障清单（建议收藏）
1. Docker Desktop 是否启动？托盘图标是否正常？
2. `docker version` 是否能看到 `Server`？
3. `docker run --rm hello-world` 是否能跑？
4. 拉取失败：检查网络/代理/镜像源/证书。
5. 挂载失败：检查 `${PWD}`、盘符路径、空格、权限。

## 进阶建议（你后续会用到）
- 构建优化：BuildKit / buildx（见 `../03-镜像/04-构建缓存与BuildKit.md`）
- Compose：多容器联调（见 `../07-Compose/`）
- 生产安全：最小权限（见 `../10-安全与最佳实践/01-最小权限原则.md`）
