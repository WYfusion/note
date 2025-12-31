# lftp 高并发文件传输工具指南

**lftp** 是一款功能强大的命令行文件传输客户端。它专为**高性能传输**设计，相比于普通的 `sftp` 或 `scp`，lftp 最大的核心优势在于其**多线程并发**能力。

在处理深度学习数据集（通常包含数十万个小文件）或下载超大模型权重时，lftp 是目前 Linux 终端下最高效的解决方案之一。

---

## 1. 核心优势与应用场景

| 特性        | 描述                     | 适用场景                          |
| :-------- | :--------------------- | :---------------------------- |
| **多线程镜像** | `mirror -P` 可并行传输多个文件。 | 上传/下载包含海量小文件的文件夹（如 ImageNet）。 |
| **分段下载**  | `pget -n` 将单个文件切片并行下载。 | 下载几十 GB 的单个模型权重文件。            |
| **断点续传**  | 进程中断后可无缝衔接继续传输。        | 网络不稳定的跨国/跨地域传输。               |
| **后台任务**  | 支持将任务放入后台队列管理。         | 长期运行的传输任务。                    |

---

## 2. 安装方法

*   **Ubuntu/Debian**:
    ```bash
    sudo apt update && sudo apt install lftp
    ```
*   **CentOS/RHEL**:
    ```bash
    sudo yum install lftp
    ```
*   **MacOS (Homebrew)**:
    ```bash
    brew install lftp
    ```

---

## 3. 连接服务器 (SFTP 模式)

虽然 lftp 支持 FTP，但在现代服务器运维中，我们主要通过 SSH 协议使用 **SFTP**。

### 3.1 交互式登录
进入 lftp 的交互式命令行界面：
```bash
# 语法：lftp sftp://user@host -p port
lftp sftp://user@192.168.1.100 -p 22
```
输入密码后，提示符变为 `lftp :~>`，此时可以输入内部命令。
```bash
lftp sftp://rtx5090@172.22.62.9 -p 2222
```

### 3.2 单行非交互登录 (脚本用)
适用于自动化脚本，直接将认证信息包含在命令中：
```bash
# 语法：lftp sftp://user:password@host:port
lftp sftp://user:123456@192.168.1.100:22
```
> **安全警告**：在命令行中明文显示密码存在安全隐患，建议使用 SSH Key 免密认证或环境变量。

---

## 4. 核心指令：Mirror (目录镜像)

`mirror` 是 lftp 最强大的指令，用于同步**整个目录**。它会自动递归处理所有子目录。

### 4.1 下载：远程 -> 本地 (Download)
将远程服务器的目录同步到本地。

```bash
# 语法：mirror [参数] 远程源目录 本地目标目录
mirror -c -P 20 --verbose remote_dataset local_dataset
```

**关键参数详解：**
*   `-c` (`--continue`)：**断点续传**。若任务中断，重启命令会跳过已完成部分，仅传输未完成的文件。
*   `-P N` (`--parallel=N`)：**并行度**（核心参数）。同时开启 N 个文件传输线程。
    *   *建议值*：小文件多时设为 `10-50`，大文件时设为 `2-5`。
*   `--verbose`：显示详细的文件传输进度。
*   `--delete`：**同步删除**。若远程删除了某文件，本地也会同步删除，保持严格一致（慎用）。

### 4.2 上传：本地 -> 远程 (Upload)
使用 `-R` (`--reverse`) 参数进行反向镜像。

```bash
# 语法：mirror -R [参数] 本地源目录 远程目标目录
mirror -R -c -P 20 local_code/ /home/user/remote_code/
```

```bash
mirror -R -c -P 20 --verbose data/NeXt_TDNN_BirdCLEF_Improved_multibank_126 /home/rtx5090/program/prenext
```
---

## 5. 核心指令：pget (大文件分段下载)

当下载**单个超大文件**（如 50GB 的 `.tar` 包）时，普通的 `get` 命令只能利用单线程。`pget` 可以将文件切分为多段同时下载，极大地利用带宽。

```bash
# 将 huge_model.tar 切成 10 段同时下载
pget -n 10 huge_model.tar
```

> **注意**：`pget` 下载过程中会生成 `huge_model.tar.lftp-pget-status` 等临时文件，下载完成后会自动合并，请勿手动删除临时文件。

---

## 6. 常用文件操作速查表

在 `lftp :~>` 终端内的常用操作：

| 命令 | 功能 | 示例 |
| :--- | :--- | :--- |
| `ls` | 列出远程文件 | `ls -lh` |
| `cd` | 切换远程目录 | `cd /data/projects` |
| `lcd` | 切换**本地**工作目录 | `lcd /home/my_pc/downloads` |
| `lpwd` | 显示**本地**当前路径 | `lpwd` |
| `get` | 下载单个文件 | `get README.md` |
| `put` | 上传单个文件 | `put main.py` |
| `mkdir` | 创建远程目录 | `mkdir new_folder` |
| `cls` | 清屏 | `cls` |
| `exit` | 退出 | `exit` |

---

## 7. 实战：一行命令模式 (One-Liner)

无需进入交互界面，直接在 Linux Shell 中执行复杂任务，非常适合集成到 `bash` 脚本中。

### 场景 A：利用 20 线程并发上传数据集
```bash
lftp -c "open sftp://user:password@192.168.1.100:22; \
mirror -R -c -P 20 /local/data/imagenet /remote/data/imagenet; \
quit"
```

### 场景 B：分段下载大模型权重
```bash
lftp -c "open sftp://user@192.168.1.100; \
pget -n 10 /remote/models/llama3-70b.tar; \
quit"
```

---

## 8. 常见问题与配置

### 8.1 解决中文乱码
如果文件名包含中文，需要在 lftp 的配置文件 `~/.lftprc` 中添加编码设置：
```bash
set file:charset utf-8
set ftp:charset utf-8
```

### 8.2 忽略 SSH Host Key 检查
在内网测试环境或频繁重装的服务器上，可能会遇到 SSH Key 变动导致无法连接。可以使用以下配置跳过确认：
```bash
lftp -e "set sftp:auto-confirm yes; open sftp://user@host..." 
```

### 8.3 书签功能
保存常用服务器，免去记忆 IP 地址：
1.  登录后输入：`bookmark add my_gpu_server`
2.  下次连接只需：`lftp my_gpu_server`