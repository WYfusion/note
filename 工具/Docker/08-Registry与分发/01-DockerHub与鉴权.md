# Docker Hub 与鉴权

## 适用场景
- 需要推送自己的镜像到 Docker Hub。
- 拉取私有镜像。
- 解决 "pull access denied" 或 "toomanyrequests" 错误。

## 1. 登录与登出 (`docker login`)
要推送镜像或拉取私有镜像，必须先登录。

```bash
# 交互式登录 (提示输入用户名和密码)
docker login

# 登录指定 Registry (如阿里云)
docker login registry.cn-hangzhou.aliyuncs.com

# 非交互式登录 (用于 CI/CD 脚本)
echo "my_password" | docker login -u my_username --password-stdin
```

> [!WARNING] 安全提示
> 尽量不要直接使用 Docker Hub 的**登录密码**。
> 推荐去 Docker Hub 官网 -> Account Settings -> Security -> **New Access Token**。
> 使用生成的 Token 作为密码登录。这样即使 Token 泄露，也可以随时撤销，且不影响主账号。

## 2. 镜像命名规范
要推送到 Docker Hub，镜像名必须包含你的**用户名**。

```bash
# 错误：没有用户名，默认推送到 library (你没权限)
docker tag my-image my-image:v1
docker push my-image:v1  # 报错: denied: requested access to the resource is denied

# 正确
docker tag my-image myusername/my-image:v1
docker push myusername/my-image:v1
```

## 3. 常见报错

### 报错 1: `denied: requested access to the resource is denied`
- **原因 A**: 你没登录 (`docker login`)。
- **原因 B**: 你尝试推送到别人的仓库 (如 `pytorch/pytorch`)，或者镜像名没加你的用户名。

### 报错 2: `toomanyrequests: You have reached your pull rate limit`
- **原因**: 匿名用户每 6 小时只能拉取 100 次，免费登录用户 200 次。
- **解决**: 执行 `docker login` 登录账号，限额会提升。

## 4. 配置文件
登录信息（加密后的 Token）保存在 `~/.docker/config.json` 中。
如果你在多台机器间迁移，可以复制这个文件（但要注意安全）。


## TODO
- [ ] 记录常见错误与解决方式
