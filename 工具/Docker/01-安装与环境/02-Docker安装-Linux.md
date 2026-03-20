# Docker 安装与配置 (Linux)

> **核心原则**：生产环境推荐使用 **Docker 官方仓库** 安装，以确保版本可控且安全。不要直接使用 `apt install docker.io` (版本通常过旧)。

## 1. Ubuntu / Debian 安装指南

### 1.1 卸载旧版本
防止版本冲突，先清理系统中的旧 Docker。
```bash
sudo apt-get remove docker docker-engine docker.io containerd runc
```

### 1.2 设置仓库 (Repository)
由于 `download.docker.com` 在国内访问极不稳定（会出现 `Could not handshake` 或 `Connection reset` 错误），**强烈建议使用阿里云镜像源**。

```bash
# 1. 更新索引并安装依赖
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg lsb-release

# 2. 添加 Docker 官方 GPG 密钥 (如果官方源卡住，可用阿里云 GPG)
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://mirrors.aliyun.com/docker-ce/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# 3. 设置阿里云镜像仓库 (解决 apt-get install 报错的核心步骤)
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://mirrors.aliyun.com/docker-ce/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```

### 1.3 安装 Docker Engine
```bash
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
```

---

## 2. CentOS / RHEL / AlmaLinux 安装指南

### 2.1 卸载旧版本
```bash
sudo yum remove docker \
                  docker-client \
                  docker-client-latest \
                  docker-common \
                  docker-latest \
                  docker-latest-logrotate \
                  docker-logrotate \
                  docker-engine
```

### 2.2 设置仓库 (使用阿里云源)
```bash
sudo yum install -y yum-utils
sudo yum-config-manager --add-repo https://mirrors.aliyun.com/docker-ce/linux/centos/docker-ce.repo
```

### 2.3 安装 Docker Engine
```bash
sudo yum install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
```

---

## 3. 服务启动与权限配置

### 3.1 启动 Docker
安装完成后，Docker 不会自动启动。
```bash
# 启动 Docker
sudo systemctl start docker

# 设置开机自启
sudo systemctl enable docker
```

### 3.2 免 Sudo 使用 Docker (重要)
默认情况下，只有 `root` 用户和 `docker` 组的用户才能访问 Docker 引擎。为了避免每次都输 `sudo`：

```bash
# 1. 创建 docker 组 (通常安装时已自动创建)
sudo groupadd docker

# 2. 将当前用户加入 docker 组
sudo usermod -aG docker $USER

# 3. 激活更改 (或者直接注销并重新登录)
newgrp docker
```
*验证：执行 `docker ps`，如果不需要 sudo 且不报错，即成功。*

---

## 4. 核心配置 (`daemon.json`)

Docker 的默认配置可能不适合生产环境（如下载慢、日志无限增长、C盘爆满）。
配置文件路径：`/etc/docker/daemon.json`。

### 4.1 一键配置命令 (命令行版)
直接复制以下命令在终端执行，即可完成**镜像加速**、**日志轮转**和**数据目录**配置。

```bash
# 创建目录
sudo mkdir -p /etc/docker

# 写入配置文件 (使用国内加速源)
sudo tee /etc/docker/daemon.json <<EOF
{
  "registry-mirrors": [
    "https://docker.m.daocloud.io",
    "https://huecker.io",
    "https://docker.nju.edu.cn",
    "https://docker.mirrors.ustc.edu.cn"
  ],
  "data-root": "/var/lib/docker",
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "100m",
    "max-file": "3"
  },
  "live-restore": true
}
EOF

# 重启 Docker 生效
sudo systemctl daemon-reload
sudo systemctl restart docker
```

#### 配置项详解
1.  **`registry-mirrors` (镜像加速)**:
    *   国内网络环境必须配置，否则拉取 `pytorch` 等大镜像会非常慢甚至失败。
2.  **`data-root` (数据目录)**:
    *   默认在 `/var/lib/docker`。
    *   ***深度学习服务器必改***：如果你的系统盘 (`/`) 很小，但有一块很大的数据盘挂载在 `/data`，请改为 `"/data/docker"`，否则跑几个模型就把系统盘撑爆了。
3.  **`log-opts` (日志轮转)**:
    *   默认 Docker 会无限记录容器的标准输出日志。
    *   限制单文件 100MB，保留 3 个文件，防止日志占满磁盘。

4.  **`live-restore`**:
    *   允许在 Docker 守护进程更新/重启时，容器继续运行（不中断服务）。

### 4.2 应用配置
修改完 `daemon.json` 后，必须重启 Docker 才能生效：
```bash
sudo systemctl daemon-reload
sudo systemctl restart docker
```

---

## 5. 验证安装

运行 Hello World 容器，确保一切正常：
```bash
docker run --rm hello-world
```
如果看到 "Hello from Docker!" 的欢迎信息，说明安装成功。
