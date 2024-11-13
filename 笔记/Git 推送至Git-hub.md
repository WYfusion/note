# Git 推送至Git-hub

### 1. 创建 GitHub 仓库

1. 登录到你的 [GitHub](https://github.com) 账号。
2. 点击右上角的 `+` 按钮，选择 **New repository**。
3. 填写仓库名称，选择是否公开或私有，然后点击 **Create repository**。

### 2. 在本地设置 Git

如果你还没有在本地安装 Git，可以根据你的操作系统下载并安装 Git：[Git 官网](https://git-scm.com/downloads)。

### 3. 克隆指定分支

```bash\
git clone -b <分支名称> <仓库URL>
```

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

### 9.使用SSH进行连接

#### 1. 生成 SSH 密钥

如果你还没有生成 SSH 密钥，需要先生成一个。在命令行（例如 Git Bash）中输入以下命令来生成 SSH 密钥：

```bash
ssh-keygen -t rsa -b 4096 -C "你的 GitHub 账户邮箱"
```

- `-t rsa` 指定使用 RSA 加密方式。
- `-b 4096` 设置密钥长度为 4096 位。
- `-C "你的 GitHub 账户邮箱"` 添加注释，一般使用 GitHub 账户的邮箱。

**提示**：

- 命令执行后会提示输入密钥的保存路径，默认路径一般是 `~/.ssh/id_rsa`，直接按 Enter 即可。
- 然后会要求设置密钥的密码，建议设置一个安全密码，直接回车则不设置密码。

这将生成两个文件：

- **私钥**：`id_rsa`（不要与他人分享）
- **公钥**：`id_rsa.pub`（需要上传到 GitHub）

#### 2. 将 SSH 公钥添加到 GitHub 账户

1. 打开生成的 

   ```
   id_rsa.pub
   ```

    文件，复制其中的内容。可以用以下命令直接在终端输出，也可以用Notepad++：

   ```bash
   cat ~/.ssh/id_rsa.pub
   ```

2. 登录 GitHub，进入“Settings”。

3. 在“Settings”页面中，找到并点击 “SSH and GPG keys”。

4. 点击 “New SSH key” 按钮。

5. 在 “Title” 栏输入该 SSH key 的名称（可以是电脑的名称或你随意定义的名称），在 “Key” 栏中粘贴刚刚复制的公钥内容。

6. 点击 “Add SSH key” 按钮保存。

#### 3. 测试 SSH 连接

为了确认 SSH 配置成功，可以使用以下命令测试与 GitHub 的连接：

```bash
ssh -T git@github.com
```

执行命令后，如果配置正确，会显示一条类似如下的欢迎消息：

```bash
Hi WYfusion! You've successfully authenticated, but GitHub does not provide shell access.
```

这表明 SSH 连接已经配置成功。

#### 4. 修改 Git 远程仓库 URL

接下来，需要将 Git 项目的远程仓库 URL 修改为 SSH 格式。

在项目的根目录下，输入以下命令：

```bash
git remote set-url origin git@github.com:WYfusion/note.git
```

此时，远程仓库的 URL 已更改为 SSH 格式。

#### 5. 使用 SSH 进行推送和拉取

完成上述步骤后，你就可以通过 SSH 来进行 Git 操作，比如推送、拉取等。

例如，执行以下命令进行推送：

```bash
git push -u origin document
```

##### 常见问题和解决方法

1. **确认 SSH-Agent 正在运行**： 如果每次操作 Git 都提示输入私钥密码，说明 SSH-Agent 未启动。可以通过以下命令启动 SSH-Agent 并添加私钥：

   ```bash
   eval "$(ssh-agent -s)"
   ssh-add ~/.ssh/id_rsa
   ```

2. **检查防火墙或网络配置**： 有些防火墙或网络配置可能会阻止 SSH 连接，尝试在不同的网络环境下进行连接测试。

```bash
31530@fusion MINGW64 /d/document/202409-202509/笔记/note/笔记 (document)
$ ssh-keygen -t rsa -b 4096 -C "fusion_wy@163.com"
Generating public/private rsa key pair.
Enter file in which to save the key (/c/Users/31530/.ssh/id_rsa):
Enter passphrase (empty for no passphrase):
Enter same passphrase again:
Your identification has been saved in /c/Users/31530/.ssh/id_rsa
Your public key has been saved in /c/Users/31530/.ssh/id_rsa.pub
The key fingerprint is:
SHA256:vGYhUAXLDsAIfUMUEG5GwpcoTwnq2VVAJblOf2lj7nc fusion_wy@163.com
The key's randomart image is:
+---[RSA 4096]----+
|*=***=*=.        |
|+*=+o+o.         |
|oo=.oo+          |
|.o+ .* .         |
| o .o + S .      |
|     . o O       |
|        O .      |
|       o . . E   |
|        ... .    |
+----[SHA256]-----+

31530@fusion MINGW64 /d/document/202409-202509/笔记/note/笔记 (document)
$ ssh -T git@github.com
Hi WYfusion! You've successfully authenticated, but GitHub does not provide shell access.

```



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

修改后再次推送：

```bash
31530@fusion MINGW64 /d/document/202409-202509/笔记/note (document)
$ cd ./笔记
31530@fusion MINGW64 /d/document/202409-202509/笔记/note/笔记 (document)
$ git add Git\ 推送至Git-hub.md
warning: in the working copy of '笔记/Git 推送至Git-hub.md', LF will be replaced by CRLF the next time Git touches it

31530@fusion MINGW64 /d/document/202409-202509/笔记/note/笔记 (document)
$ git commit -m "Git推送至Git-hub.md添加了例子"
[document c2476c8] Git推送至Git-hub.md添加了例子
 1 file changed, 106 insertions(+)

31530@fusion MINGW64 /d/document/202409-202509/笔记/note/笔记 (document)
$ git push -u origin document
Enumerating objects: 7, done.
Counting objects: 100% (7/7), done.
Delta compression using up to 16 threads
Compressing objects: 100% (4/4), done.
Writing objects: 100% (4/4), 3.03 KiB | 3.03 MiB/s, done.
Total 4 (delta 2), reused 0 (delta 0), pack-reused 0 (from 0)
remote: Resolving deltas: 100% (2/2), completed with 2 local objects.
To https://github.com/WYfusion/note.git
   99806f9..c2476c8  document -> document
branch 'document' set up to track 'origin/document'.

31530@fusion MINGW64 /d/document/202409-202509/笔记/note/笔记 (document)
$
```

注意，可以**使用Tab键自动补全**：在 `git add` 命令后输入文件名的前几个字符，然后按 Tab 键，看看 Git 是否能自动补全文件名。
