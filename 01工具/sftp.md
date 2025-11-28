# 使用 SFTP 和 sshpass 在 Ubuntu 中进行文件传输

SFTP (SSH File Transfer Protocol) 是一种安全的文件传输协议，它通过 SSH 连接提供文件访问、传输和管理功能。与 FTP 不同，SFTP 对所有数据（包括凭证和文件内容）进行加密，提供了强大的安全性。

## 1. SFTP 交互式基本用法

直接通过 `sftp` 命令进入交互式会话，手动管理文件。

### 连接远程服务器

```bash
sftp [-P port] user@hostname
```
- `-P port`: 如果服务器使用非标准端口（默认为 22），则需指定。
- `user@hostname`: 远程服务器的用户名和地址。

例如：
```bash
sftp -P 2222 a5000@172.22.62.21
```
连接成功后，您将进入 `sftp>` 提示符。

### 常用交互式命令

| 命令 | 作用 | 示例 |
| :--- | :--- | :--- |
| `ls` | 列出**远程**目录的文件。 | `ls -l` |
| `lls` | 列出**本地**目录的文件。 | `lls` |
| `cd <path>` | 切换**远程**工作目录。 | `cd /home/data` |
| `lcd <path>` | 切换**本地**工作目录。 | `lcd /mydata` |
| `pwd` | 显示**远程**当前工作目录。 | `pwd` |
| `lpwd` | 显示**本地**当前工作目录。 | `lpwd` |
| `put <local_path>` | 上传本地文件到远程。 | `put myfile.txt` |
| `get <remote_path>` | 从远程下载文件到本地。 | `get remotefile.zip` |
| `put -r <dir>` | 递归上传整个目录。 | `put -r my_folder` |
| `get -r <dir>` | 递归下载整个目录。 | `get -r remote_folder` |
| `help` | 显示帮助信息。 | `help` |
| `bye` 或 `exit` | 断开连接并退出。 | `bye` |

---

## 2. 使用 `sshpass` 实现非交互式传输

在自动化脚本中，手动输入密码是不可行的。`sshpass` 工具允许您在命令行中直接提供密码，从而实现非交互式登录。

> **注意**：`sshpass` 会将密码暴露在命令行历史中，存在安全风险。更推荐使用 **SSH 密钥认证**。

### 安装 `sshpass`

```bash
sudo apt update
sudo apt install sshpass
```

### 基本用法

`sshpass` 的基本语法是在 `sftp` 命令前加上密码。

```bash
sshpass -p 'your_password' sftp [sftp_options] user@hostname
```

### 命令详解

以下命令展示了如何结合 `sshpass` 和 `sftp` 进行连接，并处理了首次连接时的密钥验证提示。

```bash
sshpass -p wjedu123. sftp -oBatchMode=no -oStrictHostKeyChecking=no -P 2222 a5000@172.22.62.21
```

- **`sshpass -p wjedu123.`**:
  - `-p wjedu123.`: 指定密码为 `wjedu123.`。密码直接跟在 `-p` 后面。

- **`sftp [options]`**:
  - `-oBatchMode=no`:
    - `BatchMode=yes`: 批处理模式，适用于完全自动化的脚本。在此模式下，任何需要用户交互的提示（如输入密码）都会被禁止，如果出现则任务失败。
    - `BatchMode=no`: 禁用批处理模式，允许交互式提示（例如密码输入）。这里设为 `no` 是因为 `sshpass` 已经处理了密码，但保留了需要时进行其他交互的可能性。在纯脚本中，通常设为 `yes`。
  - `-oStrictHostKeyChecking=no`:
    - 首次连接新服务器时，SSH 会要求确认服务器的公钥指纹。设置为 `no` 会自动接受新主机的密钥，避免脚本因等待用户确认而中断。
    - **安全警告**：此选项会使您容易受到中间人攻击（MITM），因为它禁用了对服务器真实性的验证。仅在信任的网络环境中使用。
  - `-P 2222`: 指定远程服务器的端口号为 `2222`。
  - `a5000@172.22.62.21`: 指定用户 `a5000` 连接到 IP 地址为 `172.22.62.21` 的服务器。

---

## 3. SFTP 脚本化操作 (批处理)

对于需要执行一系列固定命令的场景（如自动上传或下载），可以使用 `sftp` 的 `-b` (batchfile) 选项。

1.  **创建一个命令文件** (`commands.txt`)：
    文件中每一行都是一个 `sftp` 命令。

    ```
    # commands.txt
    cd /remote/target/directory
    put localfile.zip
    get anotherfile.tar.gz
    ls
    ```

2.  **执行批处理命令**：

    ```bash
    sshpass -p 'your_password' sftp -oStrictHostKeyChecking=no -P 2222 -b commands.txt a5000@172.22.62.21
    ```
    - `-b commands.txt`: 指定包含 `sftp` 命令的批处理文件。
    - `sftp` 会按顺序执行文件中的所有命令，然后自动退出。

## 4. 更安全的选择：SSH 密钥认证

为了避免在脚本中明文存储密码，强烈推荐使用 SSH 密钥对进行免密认证。

1.  **在本地生成密钥对**：
    ```bash
    ssh-keygen -t rsa -b 4096
    ```

2.  **将公钥复制到远程服务器**：
    `ssh-copy-id` 是最简单的方法。
    ```bash
    ssh-copy-id -p 2222 a5000@172.22.62.21
    ```
    （此步骤需要输入一次密码）

3.  **完成！**
    之后，所有 `ssh` 和 `sftp` 连接都将不再需要密码。

    ```bash
    sftp -P 2222 a5000@172.22.62.21
    ```
