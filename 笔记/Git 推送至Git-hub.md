# Git 推送至Git-hub

### 1. 创建 GitHub 仓库

1. 登录到你的 [GitHub](https://github.com) 账号。
2. 点击右上角的 `+` 按钮，选择 **New repository**。
3. 填写仓库名称，选择是否公开或私有，然后点击 **Create repository**。

### 2. 在本地设置 Git

如果你还没有在本地安装 Git，可以根据你的操作系统下载并安装 Git：[Git 官网](https://git-scm.com/downloads)。

#### 设置 Git 全局配置

打开命令行（Terminal）或者bash并输入以下命令来配置你的 Git 用户信息：

```bash
git config --global user.name "WYfusion"
git config --global user.email "fusion_wy@163.com"
```

### 3. 初始化本地项目

在命令行中导航到你的项目目录，执行以下命令来初始化 Git 仓库：

```bash
cd /path/to/your/project  # 替换为你的项目路径
git init
```

### 4. 添加项目文件到 Git

使用以下命令将项目文件添加到 Git 暂存区：

```bash
git add .
```

### 5. 提交更改

提交更改并添加提交信息：

```bash
git commit -m "Initial commit"  # 替换为适合的提交信息
```

### 6. 连接到 GitHub 仓库

将本地仓库与 GitHub 仓库连接起来。使用以下命令（将 `your_username` 和 `your_repository` 替换为你的 GitHub 用户名和仓库名称）：

```bash
git remote add origin https://github.com/your_username/your_repository.git
```

### 7. 推送代码到 GitHub

使用以下命令将本地代码推送到 GitHub 仓库：

```bash
git push -u origin master   # 如果你在本地使用 master 分支
git push -u origin main		# 如果你在本地使用 main 分支
git pull					# 在当下也可以使用此语句，git 会自动识别当前分支
```

### 8. 验证推送

打开你的 GitHub 仓库页面，查看你的代码是否成功推送。

### 常见问题

- **如果遇到权限问题**，请确保你已正确登录 GitHub 账户，或者设置 SSH 密钥以进行身份验证。
- **如果分支名称不同**，例如使用 `main` 而不是 `master`，请相应地修改推送命令。

