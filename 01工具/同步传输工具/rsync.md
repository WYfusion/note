# rsync 高效数据同步与传输指南

**Rsync** (Remote Sync) 是业界标准的远程数据同步工具。它以**增量传输算法**闻名，能够智能识别源文件和目标文件的差异，仅传输变动的部分。

它非常适合用于代码发布、日常备份以及在本地和远程服务器之间保持文件系统的一致性。

---

## 1. 核心机制与优势

*   **增量同步 (Delta Transfer)**：
    如果一个 10GB 的文件只修改了末尾的几行，rsync 只会传输这几行的数据，而非重新发送 10GB。
*   **属性保持 (Archive Mode)**：
    能够完美保留文件的权限 (Permissions)、时间戳 (Times)、软硬链接 (Links)、所有者 (Owner) 和组 (Group)。
*   **安全性**:
    原生支持通过 SSH 隧道进行加密传输。
*   **断点续传**：
    支持在网络中断后从断点处继续传输大文件。

---

## 2. 基本语法

```bash
rsync [选项] 源路径 目标路径
```

### ⚠️ 极度重要：路径末尾的斜杠 `/`

这是 rsync 新手最容易踩的坑，请务必区分：

*   **带斜杠** (`src/`)：表示**同步目录下的内容**。
    *   命令：`rsync -a src/ dest/`
    *   结果：`dest/file1`, `dest/file2`
*   **不带斜杠** (`src`)：表示**同步目录本身**。
    *   命令：`rsync -a src dest/`
    *   结果：`dest/src/file1`, `dest/src/file2`

---

## 3. 常用参数详解

推荐组合：`-avzP` (归档+详细+压缩+进度)

| 短参 | 长参数 | 含义 | 推荐场景 |
| :--- | :--- | :--- | :--- |
| **-a** | `--archive` | **归档模式**。等同于 `-rlptgoD`。递归传输并保持所有文件属性。 | **必选** (除非你只想传文件内容不顾属性) |
| **-v** | `--verbose` | **详细输出**。显示正在传输的文件名。 | **推荐** |
| **-z** | `--compress` | **压缩传输**。在发送端压缩，接收端解压。 | 带宽低时推荐；传输已压缩文件(zip/jpg)时**不推荐** |
| **-P** | (组合参数) | 等同于 `--partial --progress`。保留部分传输的文件(断点续传)并显示进度条。 | **大文件必选** |
| **-e** | `--rsh` | 指定远程 Shell 程序，通常用于指定 SSH 端口。 | 非 22 端口必选 |
| | `--delete` | **镜像删除**。删除目标端有但源端没有的文件。 | **备份/发布时使用** (慎用) |
| | `--exclude` | **排除文件**。支持通配符，如 `*.git`。 | 排除日志、临时文件 |

---

## 4. 实战场景指南

### 场景一：本地上传到服务器 (Push)
将本地的 `project` 代码库同步到服务器，排除 `.git` 目录和 `logs` 文件夹。

```bash
rsync -avzP --exclude '.git' --exclude 'logs/' \
    -e 'ssh -p 22' \
    ./project/ user@192.168.1.100:/home/user/app/project
```
*   注意 `./project/` 带斜杠，表示将内容同步到 `/home/user/app/project` 里面。

### 场景二：从服务器下载文件 (Pull)
从服务器拉取训练好的模型权重。

```bash
rsync -avzP -e 'ssh -p 2222' \
    user@192.168.1.100:/data/models/checkpoint_v1.pt \
    ./local_models/
```

### 场景三：制作完全镜像 (慎用 --delete)
**目标**：让备份目录 `/backup` 和源目录 `/source` **一模一样**。如果 `/source` 删除了某个文件，`/backup` 也要自动删除对应文件。

```bash
rsync -avP --delete /source/ /backup/
```

> **🛑 危险警示**：
> 使用 `--delete` 前，**强烈建议**先加上 `-n` (dry-run) 参数进行空跑测试，确认不会误删重要文件！
>
> ```bash
> rsync -avP --delete -n /source/ /backup/  # 仅模拟执行，不实际删除
> ```

### 场景四：仅更新已存在的文件
如果你只想更新目标端已经存在的文件，而不传输新文件：

```bash
rsync -avP --existing source/ dest/
```

---

## 5. 进阶技巧

### 5.1 指定 SSH 端口
如果服务器 SSH 端口不是默认的 22，必须使用 `-e` 参数：
```bash
rsync -avP -e 'ssh -p 2222' source/ user@host:/dest/
```

### 5.2 限制带宽
在共享网络环境下，避免 rsync 占满带宽影响业务：
```bash
# 限制为 5MB/s (即 5000KB/s)
rsync -avP --bwlimit=5000 source/ dest/
```

### 5.3 sudo 权限问题
如果目标目录需要 root 权限才能写入，可以使用 `--rsync-path` 提权：
```bash
rsync -avP --rsync-path="sudo rsync" local_file user@host:/root/secure_dir/
```

---

## 6. 常见问题排查

1.  **Permission denied (权限被拒绝)**
    *   检查 SSH 账号是否有目标目录的写入权限 (`w`)。
    *   检查是否 SSH Key 配置正确。

2.  **传输速度慢**
    *   **CPU 瓶颈**：如果传输大量已经压缩过的文件（视频、图片），去掉 `-z` 参数。压缩过程非常消耗 CPU。
    *   **海量小文件**：rsync 默认单线程，处理数百万个小文件时效率不如 `lftp`。建议先用 `tar` 打包再传，或者换用 `lftp`。

3.  **Connection reset by peer**
    *   网络不稳定或 SSH 会话超时。加上 `-P` 参数确保可以断点续传。




```bash
sudo fpsync -n 16 -v -o "-aW --inplace" /home/a5000/data/wy/vad/ /home/a5000/Nas/3090/wy/vad/
```