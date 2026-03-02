# fpsync 笔记

## 1. 简介

`fpsync` 是一个强大的并行数据同步工具，它是 `fpart` 软件包的一部分。它本质上是一个 shell 脚本，作为 `rsync` 的包装器（wrapper）。

*   **核心功能**：利用 `fpart` 对源目录进行扫描和分块（partitioning），然后启动多个 `rsync` 进程并行地同步这些数据块。
*   **解决痛点**：标准的单进程 `rsync` 在处理包含数百万个小文件的海量目录时，扫描和传输效率较低。`fpsync` 通过并行化大大提高了同步速度，尤其是在高延迟网络或高性能存储系统上。

## 2. 工作原理

1.  **扫描与分片**：`fpsync` 调用 `fpart` 扫描源目录，根据文件数量或大小将文件列表分割成多个“分区”（partitions）。
2.  **并行传输**：`fpsync` 启动多个 `rsync` 实例，每个实例负责同步一个分区的文件列表。
3.  **任务管理**：它会监控这些 `rsync` 任务，确保它们按指定的并发数运行。

## 3. 安装

`fpsync` 通常包含在 `fpart` 软件包中。

*   **CentOS/RHEL (EPEL)**: `yum install fpart`
*   **Ubuntu/Debian**: `apt install fpart`
*   **源码编译**: 从 GitHub (https://github.com/martymac/fpart) 下载源码编译。

## 4. 基本语法

```bash
fpsync [选项] /源目录/ /目标目录/
```

**注意**：源目录和目标目录的路径末尾斜杠 `/` 的处理与 `rsync` 类似，建议源目录末尾加上 `/` 以同步目录内容而非目录本身。

## 5. 常用参数详解

| 参数 | 含义 | 详细说明 |
| :--- | :--- | :--- |
| `-n <jobs>` | **并发数** | 同时运行的 `rsync` 进程数量。默认为 2。根据 CPU 核心数和网络带宽调整，例如 `-n 8`。 |
| `-f <files>` | **每块文件数** | 每个同步任务（分区）包含的最大文件数量。默认 2000。 |
| `-s <size>` | **每块大小** | 每个同步任务（分区）包含的最大数据量。单位支持 b, k, m, g, t。默认 4GB。 |
| `-o <options>` | **rsync 参数** | 传递给底层 `rsync` 命令的参数。**非常重要**。例如 `-o "-av --delete"`。 |
| `-v` | **详细模式** | 显示更多执行信息。 |
| `-w <wrks>` | **工作目录** | 指定用于存放临时文件列表的目录。 |
| `-S` | **Sudo 模式** | 使用 sudo 运行 rsync。 |

## 6. 使用方案与示例

### 6.1. 基础本地同步

将 `/data/src/` 同步到 `/data/dst/`，使用默认参数（2个并发）。

```bash
fpsync /data/src/ /data/dst/
```

### 6.2. 提高并发数加速同步

如果有大量小文件，可以增加并发数（例如 16 个进程），并调整每个任务的文件数。

```bash
# 开启 16 个并发进程
fpsync -n 16 -v /data/src/ /data/dst/
```

### 6.3. 传递 rsync 参数（归档模式并删除差异）

这是最常用的场景。我们需要保留权限、时间戳（`-av`），并且删除目标端有但源端没有的文件（`--delete`）。

```bash
# -o 参数后的内容必须用引号括起来
fpsync -n 8 -o "-av --delete" /data/src/ /data/dst/
```

### 6.4. 远程同步（通过 SSH）

`fpsync` 也支持通过 SSH 同步到远程服务器。目标路径格式为 `user@host:/path/`。

**注意**：为了避免每次都要输入密码，建议配置 SSH 密钥免密登录。

```bash
# 将本地 /data/src/ 同步到远程服务器 192.168.1.100 的 /backup/dst/
# 启用数据压缩 (-z)
fpsync -n 4 -o "-avz" /data/src/ root@192.168.1.100:/backup/dst/
```

拉取到本地
```bash
fpsync -n 16 -v -o "-W --inplace --size-only" "/home/gzhu/Nas/黄茅海数据/2025/3-5/" /home/gzhu/data4/2025/3-5/
```

#### 查看磁盘负载
```bash
iostat -dx 1
```

#### 使用示例
```bash
fpsync -n 128 -v -o "-av" /home/gzhu/data3/cutstandardrename/W9/ rtx5090_target:/home/rtx5090/data/cut/W9
```

```bash
fpsync -n 48 -v -o "-lptgoD -v --numeric-ids -u" /home/gzhu/data1/wy/Data_processing/experiment/xeno-ebird0.5h-AU-se_vad-nobug/noisy_wav/ 5090:/home/rtx5090/data/wy/SEtrain/exp/nohave/
```

### 6.5. 针对海量小文件的优化

对于包含数亿小文件的目录，文件列表的生成本身就很耗时。可以调整 `-f` 和 `-s` 来平衡分片大小。

```bash
# 每个任务处理 50000 个文件或 1GB 数据，并发 32
fpsync -n 32 -f 50000 -s 1G -o "-av" /src/ /dst/
```

### 6.6. 断点续传与中断恢复

在任务中断（如网络断开或手动停止）后再次运行 `fpsync`，恢复机制取决于文件状态：

1.  **已传输完成的文件（小文件场景）**：
    *   **不需要重传**。`rsync` 会自动比对源和目标文件（默认检查大小和修改时间），如果目标文件已存在且一致，会直接跳过。
    *   无论是否使用 `--partial`，这一点都成立。因此对于海量小文件，中断后直接重跑即可。

2.  **正在传输中的文件（大文件场景）**：
    *   **未使用 `--partial`**：`rsync` 默认会**删除**传输了一半的临时文件。再次运行时，该文件必须**从头开始重传**。
    *   **使用了 `--partial`**：保留半成品文件。再次运行时，`rsync` 会利用已传输的部分，实现真正的“文件级断点续传”。

```bash
# 建议始终添加 --partial 以防止大文件传输中断导致的前功尽弃
fpsync -n 8 -o "-av --partial" /data/src/ /data/dst/
```

## 7. 最佳实践与注意事项

1.  **免密登录**：进行远程同步时，务必配置 SSH Key 免密登录，否则每个并发的 rsync 进程都会提示输入密码。
2.  **资源监控**：并发数 `-n` 不是越大越好。过多的并发会导致磁盘 I/O 瓶颈或 CPU 上下文切换开销，反而降低速度。建议从 CPU 核心数的 1-2 倍开始测试。
3.  **临时目录**：`fpsync` 会在 `/tmp` 下生成大量临时文件列表。如果 `/tmp` 空间不足，使用 `-w` 指定其他目录。
4.  **最后一次同步**：`fpsync` 主要是为了加速数据传输。在最终切换业务前，建议运行一次标准的单进程 `rsync -av --delete` 进行最后的一致性校验，确保没有遗漏（虽然 fpsync 很可靠，但单进程 rsync 是最终一致性的黄金标准）。
