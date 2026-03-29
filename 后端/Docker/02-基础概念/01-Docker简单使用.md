# Docker 简单使用（从 0 到能干活）

> 目标：用最少的命令完成“拉镜像 → 跑容器 → 进容器排查 → 看日志 → 导出/清理”的闭环。

## 适用场景
- 刚装好 Docker，想验证是否可用并开始跑服务。
- 日常开发/联调，需要快速启动一个依赖（nginx、redis、postgres、jupyter…）。
- 容器出问题时，知道用哪些命令定位。

## Windows vs Linux 差异速查（很重要）
> 你可能同时在 Windows 和 Linux 用 Docker。下面这些差异决定了你“同一条命令为什么在另一台机器不灵”。

### 1) Docker daemon 在哪里跑？
- **Linux 原生**：daemon（`dockerd`）就在本机 Linux 上。
- **Windows + Docker Desktop**：大多数情况下 daemon 实际跑在 **WSL2 的 Linux 环境**（或一个轻量 VM）里。

影响：
- `docker` CLI 的命令形式基本一致，但“网络/文件系统/证书/代理”的行为可能不同。

### 2) 路径挂载写法不同
- Linux：`-v /abs/path:/container/path`
- Windows PowerShell：推荐用 **`${PWD}`** 或明确盘符路径（注意引号）
- Windows CMD：`%cd%`

### 3) 权限与文件属性
- Linux：容器内的 UID/GID、可执行权限位更“真实”；挂载时权限问题更常见。
- Windows：挂载 NTFS 时可能遇到权限/大小写/换行（CRLF）带来的差异。

### 4) 网络与防火墙
- Linux：端口映射通常比较直观，iptables/NAT 排障常见。
- Windows：除了 Docker 端口映射外，还可能受 Windows 防火墙/企业安全策略影响。

## 先记住 3 个核心对象
1. **Image（镜像）**：模板、只读、可复用。
2. **Container（容器）**：镜像的运行实例，有自己的可写层。
3. **Registry（仓库）**：镜像的“云端存放处”（Docker Hub / 私有仓库）。

## 最小闭环：验证安装是否可用
### 1) 看版本与环境

```bash
docker version
docker info
```

重点：`docker version` 必须能看到 **Server**。

### 2) 跑一个 hello-world

```bash
docker run --rm hello-world
```

解释：
- `run`：创建并启动容器
- `--rm`：退出后自动删除容器（避免垃圾堆积）

## 镜像：拉取、查看、打标签
### 1) 拉取镜像：`docker pull`

```bash
docker pull nginx:latest
```

注意：
- `latest` 不等于“最新”，更不等于“稳定”。生产建议固定版本号（见 `../03-镜像/05-镜像标签与版本策略.md`）。

### 2) 查看本地镜像：`docker images`

```bash
docker images
```

### 3) 给镜像打标签：`docker tag`

```bash
docker tag nginx:latest my-nginx:v1
```

### 4) 删除镜像：`docker rmi`

```bash
docker rmi my-nginx:v1
```

## 容器：启动一个服务（最常用参数讲透）
> `docker run` 是 80% 场景的核心。

### 1) 后台启动：`-d`

```bash
docker run -d nginx
```

### 2) 命名：`--name`

```bash
docker run -d --name web nginx
```

好处：后续用名字代替容器 ID（更好记）。

### 3) 端口映射：`-p host:container`
把容器 80 映射到宿主机 8080：

```bash
docker run -d --name web -p 8080:80 nginx
```

只允许本机访问（更安全）：

```bash
docker run -d --name web -p 127.0.0.1:8080:80 nginx
```

### 4) 目录挂载：`-v`（或 `--mount`）
把宿主机目录挂到容器里：

```bash
docker run --rm -it -v /path/on/host:/workspace ubuntu bash
```

Windows 注意：
- PowerShell 使用 **`${PWD}`**：

```powershell
docker run --rm -it -v ${PWD}:/workspace ubuntu bash
```

补充：
- 如果你在 **WSL2 的 Linux 终端**里执行 docker（而不是 PowerShell），那挂载路径就按 Linux 写：

```bash
docker run --rm -it -v $(pwd):/workspace ubuntu bash
```

- Windows 挂载的更多坑位（空格、盘符、CRLF、权限）见：`../06-存储/03-Windows路径挂载注意事项.md`。

### 5) 环境变量：`-e`

```bash
docker run --rm -e APP_ENV=dev ubuntu env
```

### 6) 交互式终端：`-it`

```bash
docker run --rm -it ubuntu bash
```

解释：
- `-i`：保持 STDIN 打开
- `-t`：分配伪终端

## 容器管理：看、进、停、删
### 1) 查看容器

```bash
docker ps
docker ps -a
```

### 2) 查看日志：`docker logs`

```bash
docker logs web
docker logs -f --tail 200 web
```

参数解释：
- `-f`：持续跟随
- `--tail 200`：只看最后 200 行（排障很常用）

### 3) 进入容器：`docker exec`

```bash
docker exec -it web bash
```

注意：
- `exec` 进入的是“正在运行”的容器。
- 端口映射必须在 `docker run/create` 时做，**不能**通过 `exec` 添加。

Windows/WSL2 补充：
- 在 Windows 上 `docker exec -it xxx bash` 失败的常见原因是镜像里没有 bash（例如 alpine）。这种情况下换成：
	- `sh`

### 4) 查看细节：`docker inspect`

```bash
docker inspect web
```

常用：看 IP、挂载、环境变量、入口命令。

### 5) 查看资源：`docker stats`

```bash
docker stats
docker stats web
```

### 6) 停止 / 启动 / 重启

```bash
docker stop web
docker start web
docker restart web
```

### 7) 删除容器

```bash
docker rm web
docker rm -f web
```

## 文件拷贝：`docker cp`（排障/导出很实用）
从容器复制到宿主机：

```bash
docker cp web:/etc/nginx/nginx.conf ./nginx.conf
```

从宿主机复制到容器：

```bash
docker cp ./index.html web:/usr/share/nginx/html/index.html
```

## 常见“参数组合模板”（直接套用）
### 1) 临时调试容器

```bash
docker run --rm -it ubuntu bash
```

### 2) 启动服务（后台 + 命名 + 端口）

```bash
docker run -d --name web -p 8080:80 nginx
```

### 3) 开发挂载（代码目录映射进去）

```bash
docker run --rm -it -v /path/to/code:/workspace -w /workspace ubuntu bash
```

参数解释：
- `-w /workspace`：设置工作目录（比你进去后再 cd 更省事）

## 清理（避免磁盘爆炸）
### 1) 看占用

```bash
docker system df
```

### 2) 清理无用资源（谨慎）
更完整内容见：`../04-容器/06-清理与磁盘占用治理.md`。

## 常见坑位（最容易踩）
1. **端口映射方向搞反**：`-p 宿主机端口:容器端口`。
2. **容器内服务监听 127.0.0.1**：即使映射端口也访问不到（服务必须监听 0.0.0.0）。
3. **挂载路径写错**：PowerShell 用 `${PWD}`；路径含空格要注意引号。
4. **忘了 --rm**：临时容器退出后堆积，磁盘慢慢炸。

## 平台相关排障小抄（很实用）
### A) 同一条 `-p 8080:80` 在 Windows 上别人访问不了
优先检查：
- 你是否用 `127.0.0.1:8080:80` 绑定了仅本机访问
- Windows 防火墙是否拦截了该端口

### B) 挂载目录后容器里看不到改动/权限异常
优先检查：
- 你是在 PowerShell 还是 WSL2 里执行 docker？（路径写法不同）
- 路径是否指向了正确位置（尤其是多 repo、多终端时）
- Windows 文本换行 CRLF 是否导致 shell/python 脚本不可执行

### C) Linux 上容器读写权限不对
优先检查：
- 容器内进程运行用户（`--user`）
- 宿主机目录属主/权限（UID/GID）


## 下一步建议
- 写 Dockerfile：见 `../03-镜像/02-Dockerfile语法速查.md`
- 多容器联调：见 `../07-Compose/01-DockerCompose快速上手.md`
- 了解资源限制：见 `03-命名空间与Cgroups速记.md`
