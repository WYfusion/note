# SFTP 安全文件传输与自动化指南

**SFTP** (SSH File Transfer Protocol) 是一种运行在 **SSH** (Secure Shell) 协议之上的安全文件传输协议。
与传统的 FTP 不同，SFTP 对所有的传输命令和数据进行加密，能有效防止密码窃听和中间人攻击。

---

## 1. 交互式模式 (基础用法)

这是最常用的手动管理文件的方式，操作逻辑与 Linux Shell 类似。

### 1.1 建立连接
```bash
# 默认端口 22
sftp user@hostname

# 指定端口 (注意是大写 -P)
sftp -P 2222 user@192.168.1.100
```

### 1.2 常用内部命令速查

| 命令 | 作用 | 示例 |
| :--- | :--- | :--- |
| `pwd` / `lpwd` | 显示**远程** / **本地** 当前路径 | `pwd` |
| `ls` / `lls` | 列出**远程** / **本地** 文件列表 | `ls -la` |
| `cd` / `lcd` | 切换**远程** / **本地** 目录 | `cd /var/www` |
| `put` | **上传**文件 (本地 -> 远程) | `put file.txt` |
| `put -r` | **递归上传**文件夹 | `put -r ./data_folder` |
| `get` | **下载**文件 (远程 -> 本地) | `get file.txt` |
| `get -r` | **递归下载**文件夹 | `get -r ./remote_folder` |
| `bye` | 退出 | `bye` |

---

## 2. 自动化与非交互式传输

在编写 Shell 脚本（如每日自动备份）时，我们需要绕过手动输入密码的交互环节。

### 方法 A：SSH 密钥认证 (推荐，最安全) ✨

这是生产环境的最佳实践。通过配置 SSH 公钥/私钥，实现完全免密登录。

1.  **生成密钥对** (如果还没有的话)：
    ```bash
    ssh-keygen -t rsa -b 4096
    ```
2.  **将公钥上传到服务器**：
    ```bash
    ssh-copy-id -p 2222 user@192.168.1.100
    ```
3.  **免密登录**：
    配置完成后，直接执行 `sftp` 将不再询问密码：
    ```bash
    sftp -P 2222 user@192.168.1.100
    ```

### 方法 B：使用 `sshpass` (不推荐，仅用于特定场景)

如果无法配置 SSH Key（例如没有服务器管理权限），可以使用 `sshpass` 工具直接在命令行传入密码。

> **⚠️ 安全风险**：密码会以明文形式出现在历史记录 (`history`) 和进程列表 (`ps`) 中！

1.  **安装工具**：
    ```bash
    sudo apt install sshpass  # Ubuntu/Debian
    sudo yum install sshpass  # CentOS
    ```

2.  **单行命令示例**：
    ```bash
    # 语法：sshpass -p '密码' sftp [参数] user@host
    
    # 示例：自动登录并下载文件
    sshpass -p 'MySecretPass' sftp -oBatchMode=no -P 22 user@192.168.1.100 <<< "get remote_file.txt"
    sshpass -p 'your_password' sftp -oStrictHostKeyChecking=no -P 2222 a5000@172.22.62.21
	sshpass -p 'your_password' sftp -oStrictHostKeyChecking=no -P 2222 -b commands.txt a5000@172.22.62.21
    ```

---

## 3. 批处理模式 (Batch Mode)

当你需要一次性执行多个操作（例如：进入目录 -> 上传A -> 上传B -> 退出）时，可以使用批处理文件。

### 步骤 1：编写脚本文件
创建一个名为 `batch_script.txt` 的文本文件，写入 SFTP 命令：

```text
cd /data/backups # 1. 切换远程目录
lcd /local/data  # 2. 切换本地目录
put -r database_dump_2023/  # 3. 上传文件
rename access.log access.log.bak  # 4. 重命名文件
bye  # 5. 退出连接
```

在 sftp 命令中，-b 参数后面跟的“脚本”并不是我们常见的 Shell 脚本（如 .sh）或 Python 脚本，而是一个纯文本文件，里面包含的是 SFTP 内部命令列表。简单来说，-b 的含义是 Batch Mode（批处理模式）。  它的作用是告诉 SFTP 客户端：“不要等待我用键盘输入命令，而是读取这个文件里的每一行指令，并按顺序自动执行。”
  以下是关于 -b 参数的详细严谨解释：
  1. 这里的“脚本”到底是什么？
   * 格式：普通的 .txt 纯文本文件。
   * 内容：每一行都是一个你平时在 sftp> 提示符后手动输入的命令（如 cd, put, get, ls, mkdir 等）。
   * 限制：
       * 不能包含 Linux Shell 命令（如 grep, awk, git 等，除非 sftp 服务器支持 !command 语法，但通常不推荐）。
       * 没有逻辑控制（不支持 if, for, while 等编程逻辑）。它只是傻瓜式地从第一行执行到最后一行。

  2. 举个具体的例子

  假设你每天都要把本地的日志上传到服务器，手动操作太麻烦。你可以创建一个名为 upload_task.txt 的文件：
  A. 错误处理机制
  默认情况下，如果在批处理执行过程中任何一条命令失败（例如目录不存在、权限不足），SFTP 会立即终止连接并报错退出。
   * 这是一种保护机制，防止后续命令在错误的状态下继续执行（比如没切进目录就开始删文件）。
  B. 必须配合免密认证
  -b 只是解决了“命令输入”的自动化，没有解决“密码输入”的自动化。
   * 如果你运行 sftp -b commands.txt user@host 但没有配置 SSH Key 免密，程序会卡住等待你输入密码，这就失去了自动化的意义。
   * 所以 -b 通常配合 SSH Key 或 sshpass 一起使用。

  C. 与 Shell 脚本的区别
   * Shell 脚本 (`run.sh`)：运行在你的本地电脑上，可以做逻辑判断、循环、调用各种软件。
   * SFTP 批处理文件 (`cmd.txt`)：仅仅是传给 SFTP 程序的“代办事项清单”，仅限于文件传输操作。


### 步骤 2：执行批处理
使用 `-b` 参数加载脚本：

```bash
# 使用 SSH Key 免密认证
sftp -P 22 -b batch_script.txt user@hostname

# 或者结合 sshpass
sshpass -p 'pass' sftp -P 22 -b batch_script.txt user@hostname
```

---

## 4. 关键参数说明

在脚本中为了保证稳定性，通常会加上以下 SSH 选项：

```bash
sftp -oBatchMode=no -oStrictHostKeyChecking=no user@host
```

*   `-oBatchMode=no`:
    *   在配合 `sshpass` 时必须设为 `no`。如果设为 `yes`，SSH 会直接尝试密钥登录，如果失败则立即报错，不会给 `sshpass` 输入密码的机会。
*   `-oStrictHostKeyChecking=no`:
    *   **功能**：自动接受新的主机指纹，不弹出 "Are you sure you want to continue connecting (yes/no)?" 的提示。
    *   **场景**：适用于自动化脚本，防止因主机指纹变更或首次连接导致的脚本卡死。
    *   **风险**：降低了安全性，可能面临中间人攻击风险。

---

## 5. 总结：我该选哪种？

| 需求场景 | 推荐方案 | 关键技术 |
| :--- | :--- | :--- |
| **偶尔手动传文件** | 交互式 SFTP | `put`, `get` |
| **定时备份脚本** | SSH 密钥认证 + 批处理 | `ssh-copy-id`, `sftp -b` |
| **无法配置密钥** | sshpass (慎用) | `sshpass` |
| **传输海量文件** | **不推荐 SFTP** | 请改用 **rsync** 或 **lftp** |