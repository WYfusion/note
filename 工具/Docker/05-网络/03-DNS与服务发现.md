# DNS 与服务发现

## 适用场景
- **pip install 失败**：报错 `Could not resolve host: pypi.org`。
- **代码报错**：连接数据库时报错 `Name or service not known`。
- **微服务调试**：想知道为什么 `ping mysql` 通，但 `ping www.baidu.com` 不通。

## 1. Docker 的 DNS 机制
Docker 容器并没有“真正的”独立 DNS 服务，它通过挂载和拦截来实现域名解析。

### 1.1 默认行为
容器启动时，Docker 会把宿主机的 `/etc/resolv.conf` 复制进容器。
- **如果宿主机能上网**，容器通常也能解析域名。
- **例外**：如果宿主机用的是 `127.0.0.53` (systemd-resolved) 这种本地 DNS 存根，Docker 会自动将其替换为 Google 的 `8.8.8.8`，防止容器解析失败。

### 1.2 内部服务发现 (Embedded DNS)
当你使用**自定义网络** (`docker network create`) 时，Docker 会启动一个内置 DNS 服务器（在 `127.0.0.11`）。
- 它负责解析**容器名**（如 `ping redis` -> `172.18.0.2`）。
- 解析不了的外部域名，转发给上游 DNS。

## 2. 常见 DNS 故障与修复

### 故障 1：无法解析外部域名 (pip/apt 失败)
**现象**：`ping 8.8.8.8` 通，但 `ping www.baidu.com` 报错 `Temporary failure in name resolution`。
**原因**：宿主机的 DNS 配置在容器内不可用（例如公司内网 DNS 无法被容器访问，或者被防火墙拦截）。
**解决**：强制指定 DNS。
1. **临时 (docker run)**:
   ```bash
   docker run --dns 8.8.8.8 --dns 114.114.114.114 ...
   ```
2. **永久 (daemon.json)**:
   修改 `/etc/docker/daemon.json`，让所有容器默认使用指定 DNS：
   ```json
   {
     "dns": ["114.114.114.114", "8.8.8.8"]
   }
   ```

### 故障 2：无法解析容器名
**现象**：`ping mysql` 报错 `Name or service not known`。
**原因**：你可能在使用默认的 `bridge` 网络。
**解决**：**默认 bridge 网络不支持 DNS 服务发现**。必须创建自定义网络。
```bash
# 错误
docker run -d --name db redis
docker run -it --link db:db ... (过时写法)

# 正确
docker network create my-net
docker run -d --name db --network my-net redis
docker run -it --network my-net busybox ping db
```

## 3. 调试命令
在容器内排查 DNS 问题：

```bash
# 1. 检查 DNS 配置
cat /etc/resolv.conf
# 正常应该看到 nameserver 127.0.0.11 (自定义网络) 或 宿主机DNS

# 2. 测试解析
nslookup www.baidu.com
dig mysql
```

> [!TIP] 深度学习环境建议
> 如果你在公司内网，且 pip 源是内网地址（如 `pypi.corp.com`），务必确保容器能使用公司的 DNS 服务器，否则只能通过 IP 访问，非常麻烦。

