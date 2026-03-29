# 私有仓库 Registry 搭建

## 适用场景
- **内网环境**：服务器无法访问外网，需要一个内部的镜像中心。
- **速度优化**：局域网拉取镜像比从 Docker Hub 拉取快得多。
- **隐私安全**：不想把核心算法镜像传到公网。

## 1. 极简搭建 (HTTP 模式)
使用官方提供的 `registry:2` 镜像。

```bash
docker run -d \
  -p 5000:5000 \
  --restart=always \
  --name registry \
  -v /mnt/registry:/var/lib/registry \
  registry:2
```
*数据会持久化在宿主机的 `/mnt/registry` 目录下。*

## 2. 推送镜像到私有仓库

### 步骤 1: 打标签 (Tag)
必须把镜像名改成 `IP:端口/镜像名` 的格式。
```bash
# 假设私有仓库 IP 是 192.168.1.100
docker tag my-app:v1 192.168.1.100:5000/my-app:v1
```

### 步骤 2: 配置“不安全仓库”信任
默认 Docker 强制使用 HTTPS。如果是 HTTP 私有仓库，必须在**所有客户端**配置白名单。
修改 `/etc/docker/daemon.json`:
```json
{
  "insecure-registries": ["192.168.1.100:5000"]
}
```
重启 Docker: `systemctl restart docker`。

### 步骤 3: 推送 (Push)
```bash
docker push 192.168.1.100:5000/my-app:v1
```

### 步骤 4: 拉取 (Pull)
在另一台机器上（也配置了 insecure-registries 后）：
```bash
docker pull 192.168.1.100:5000/my-app:v1
```

## 3. 进阶：带密码认证 (Basic Auth)
默认 registry 是公开的，谁都能推。生产环境需要加密码。

1. **生成密码文件**:
   ```bash
   mkdir auth
   docker run --entrypoint htpasswd registry:2 -Bbn myuser mypassword > auth/htpasswd
   ```

2. **启动带认证的 Registry**:
   ```bash
   docker run -d \
     -p 5000:5000 \
     --restart=always \
     --name registry-auth \
     -v $(pwd)/auth:/auth \
     -e "REGISTRY_AUTH=htpasswd" \
     -e "REGISTRY_AUTH_HTPASSWD_REALM=Registry Realm" \
     -e "REGISTRY_AUTH_HTPASSWD_PATH=/auth/htpasswd" \
     -v /mnt/registry:/var/lib/registry \
     registry:2
   ```

3. **登录**:
   ```bash
   docker login 192.168.1.100:5000
   # 输入 myuser / mypassword
   ```


## TODO
- [ ] 给出 compose 方式部署示例
- [ ] 常见坑：证书、insecure registry
