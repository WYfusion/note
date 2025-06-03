养成实现一个新功能在需要在新的分支中实现的好习惯，避免分支冲突
```bash
git version
```

```bash
git config --global user.name "WYfusion"
```

```bash
git config --global user.email "fusion_wy@163.com"
```

##### 创建ssh连接密钥公钥
作用：使用ssh克隆git-hub上的项目

#### 1. 生成 SSH 密钥packet-beta
title UDP Packet
0-15: "Source Port"
16-31: "Destination Port"
32-47: "Length"
48-63: "Checksum"
64-95: "Data (variable length)"



如果你还没有生成 SSH 密钥，需要先生成一个。在命令行（例如 Git Bash）中输入以下命令来生成 SSH 密钥：

```bash
ssh-keygen -t rsa -b 4096 -C "fusion_wy@163.com"
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

   ```bash
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
7. 注意若是改变了默认的名称，需要在`~/.ssh/config`中添加下面的内容
```shell
Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_git_rsa
    IdentitiesOnly yes
```

再对这些文件授予合适权限
```bash
chmod 600 ~/.ssh/id_git_rsa      # 私钥权限
chmod 644 ~/.ssh/id_git_rsa.pub  # 公钥权限
chmod 600 ~/.ssh/config          # 配置文件权限
```

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


##### ==注意不同账户下进行版本控制需要将目录送入安全模式==

```bash
git config --global --add safe.directory D:/document/202409-202509/code/pythonProject2
```

`.gitignore` 文件下添加路径会在添加文件时忽略这里的路径

```bash
git init
```

```bash
git add test.py
```

```bash
git add .
```

```bash
git checkout -b main # 创建并进入main分支
```

```bash
git branch main # 创建main分支
```

```bash
git branch | cat # 检查所有分支
```

```bash
git checkout main # 进入main分支
```

```bash
git switch main # 进入main分支
```

```shell
git reset HEAD # 撤销所有暂存区的修改 撤销git add误添加的文件
```

```shell
git reset HEAD file.txt # 仅撤销其中某个
```

```bash
git commit -m "第二次提交"
```

```bash
git log
```

```bash
git checkout -b new-branch-name 7be9d50970bc6515bd2160f96547fedac03fb37
# 依据指定节点创建分支
```

``` bash
git branch  # 列出来所有分支
```

```bash
git branch -d <branch_name>
```

``` bash
git branch -D <branch_name>
```

``` bash
git branch -d feature-1 feature-2
```

``` bash
git branch -D feature-1 feature-2 # 强制删除多个
```

```bash
git push origin --delete <branch_name> 
```

```bash
git branch -m <new_branch_name>  # 改分支名
```

```bash
git merge feature-branch  # 将feature-branch中的最新提交内容合并到当前分支中
```

####  使用 SSH 进行推送和拉取

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


##### 忽略指定文件夹
创建`.gitignore`隐藏文件
在其中添加欲隐藏的文件夹即可，注意这会忽略项目中所有名为所列文件名的文件夹(无论在哪个文件夹下)
```shell
data/
dataset/
logs/
.obsidian/
```
