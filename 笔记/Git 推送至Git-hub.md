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

## 例子：

```bash
31530@fusion MINGW64 /d/document/202409-202509/笔记
$ git config --global user.name "WYfusion"       # 设置全局用户变量
git config --global user.email "fusion_wy@163.com"

31530@fusion MINGW64 /d/document/202409-202509/笔记
$ git clone https://github.com/WYfusion/note.git  # 克隆远程仓库，这样可以避免出现两个不相关的历史记录。这通常发生在尝试合并或拉取一个与当前本地分支没有共同祖先的远程分支时。
Cloning into 'note'...
remote: Enumerating objects: 4, done.
remote: Counting objects: 100% (4/4), done.
remote: Compressing objects: 100% (3/3), done.
remote: Total 4 (delta 0), reused 0 (delta 0), pack-reused 0 (from 0)
Receiving objects: 100% (4/4), 12.76 KiB | 933.00 KiB/s, done.

31530@fusion MINGW64 /d/document/202409-202509/笔记
$ cd ./note

31530@fusion MINGW64 /d/document/202409-202509/笔记/note (main)
$ git add .
warning: in the working copy of '笔记/Anaconda.md', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of '笔记/BN(BatchNormalization)BN(Batch Normalization)原理.md', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of '笔记/Bash(Bourne Again Shell)Bash(Bourne Again Shell).md', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of '笔记/Git 推送至Git-hub.md', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of '笔记/Git.md', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of '笔记/Google colab.md', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of '笔记/Jupyter.md', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of '笔记/Tensorboard.md', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of '笔记/卷积与转置卷积.md', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of '笔记/参数的优化器.md', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of '笔记/感受野计算.md', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of '笔记/迁移学习.md', LF will be replaced by CRLF the next time Git touches it

31530@fusion MINGW64 /d/document/202409-202509/笔记/note (main)
$ git commit -m "2024年11月9日的提交"
[main 99806f9] 2024年11月9日的提交
 28 files changed, 1420 insertions(+)
 create mode 100644 "\347\254\224\350\256\260/Anaconda.md"
 create mode 100644 "\347\254\224\350\256\260/BN(BatchNormalization)BN(Batch Normalization)\345\216\237\347\220\206.md"
 create mode 100644 "\347\254\224\350\256\260/Bash(Bourne Again Shell)Bash(Bourne Again Shell).md"
 create mode 100644 "\347\254\224\350\256\260/Git \346\216\250\351\200\201\350\207\263Git-hub.md"
 create mode 100644 "\347\254\224\350\256\260/Git.md"
 create mode 100644 "\347\254\224\350\256\260/Google colab.md"
 create mode 100644 "\347\254\224\350\256\260/Jupyter.md"
 create mode 100644 "\347\254\224\350\256\260/Tensorboard.md"
 create mode 100644 "\347\254\224\350\256\260/assets/image-20241015210459378.png"
 create mode 100644 "\347\254\224\350\256\260/assets/image-20241020131207935.png"
 create mode 100644 "\347\254\224\350\256\260/assets/image-20241022193902553.png"
 create mode 100644 "\347\254\224\350\256\260/assets/image-20241023093157085.png"
 create mode 100644 "\347\254\224\350\256\260/assets/image-20241023094107063.png"
 create mode 100644 "\347\254\224\350\256\260/assets/image-20241023094846623.png"
 create mode 100644 "\347\254\224\350\256\260/assets/image-20241023094904259.png"
 create mode 100644 "\347\254\224\350\256\260/assets/image-20241023095248331.png"
 create mode 100644 "\347\254\224\350\256\260/assets/image-20241023100244301.png"
 create mode 100644 "\347\254\224\350\256\260/assets/image-20241023134154676.png"
 create mode 100644 "\347\254\224\350\256\260/assets/image-20241023134202461.png"
 create mode 100644 "\347\254\224\350\256\260/assets/image-20241023134435581.png"
 create mode 100644 "\347\254\224\350\256\260/assets/image-20241023161433781.png"
 create mode 100644 "\347\254\224\350\256\260/assets/\345\215\267\347\247\257\345\216\237\347\220\206\345\233\276-1729410767684-4.png"
 create mode 100644 "\347\254\224\350\256\260/\344\270\215\347\206\237\346\202\211\347\232\204\351\203\250\345\210\206.md"
 create mode 100644 "\347\254\224\350\256\260/\345\215\267\347\247\257\344\270\216\350\275\254\347\275\256\345\215\267\347\247\257.md"
 create mode 100644 "\347\254\224\350\256\260/\345\217\202\346\225\260\347\232\204\344\274\230\345\214\226\345\231\250.md"
 create mode 100644 "\347\254\224\350\256\260/\345\244\232GPU\345\271\266\350\241\214\350\256\241\347\256\227.md"
 create mode 100644 "\347\254\224\350\256\260/\346\204\237\345\217\227\351\207\216\350\256\241\347\256\227.md"
 create mode 100644 "\347\254\224\350\256\260/\350\277\201\347\247\273\345\255\246\344\271\240.md"

31530@fusion MINGW64 /d/document/202409-202509/笔记/note (main)
$ git status  			# 查看到有一次提交状态更新信息
On branch main
Your branch is ahead of 'origin/main' by 1 commit.
  (use "git push" to publish your local commits)

nothing to commit, working tree clean

31530@fusion MINGW64 /d/document/202409-202509/笔记/note (main)
$ git checkout -b document  # 创建并切换到指定的分支
Switched to a new branch 'document'

31530@fusion MINGW64 /d/document/202409-202509/笔记/note (document)
$ git remote add origin https://github.com/WYfusion/note.git # 由于已经在上面使用了git clone所以不需要再次链接到远程仓库了
error: remote origin already exists.

31530@fusion MINGW64 /d/document/202409-202509/笔记/note (document)
$ git remote -v  										     # 检验当前的远程仓库 
origin  https://github.com/WYfusion/note.git (fetch)
origin  https://github.com/WYfusion/note.git (push)

31530@fusion MINGW64 /d/document/202409-202509/笔记/note (document)
$ git push -u origin document								 # 将本地的document分支提交到远程的document中去
Enumerating objects: 31, done.
Counting objects: 100% (31/31), done.
Delta compression using up to 16 threads
Compressing objects: 100% (30/30), done.
Writing objects: 100% (30/30), 707.57 KiB | 28.30 MiB/s, done.
Total 30 (delta 1), reused 0 (delta 0), pack-reused 0 (from 0)
remote: Resolving deltas: 100% (1/1), done.
To https://github.com/WYfusion/note.git
   730dbe0..99806f9  document -> document
branch 'document' set up to track 'origin/document'.

31530@fusion MINGW64 /d/document/202409-202509/笔记/note (document)
$

```

