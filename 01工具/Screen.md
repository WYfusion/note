# Screen 命令详细指南

GNU Screen 是一个强大的终端多路复用器，它允许用户在一个终端会话中运行多个独立的终端程序，并在它们之间自由切换。即使网络连接中断，Screen 会话也能保持运行，确保您的任务不会中断。

## 1. 安装 Screen

根据您的操作系统选择相应的安装命令：

```bash
# Debian/Ubuntu 系统
sudo apt update
sudo apt install screen
```

```bash
# CentOS/RHEL 系统
sudo yum install screen
```

## 2. 基本操作

### 2.1. 新建会话

*   **创建匿名会话**:
    ```bash
    screen
    ```
*   **创建命名会话**:
    ```bash
    screen -S <session_name>
    ```
    例如：`screen -S my_session`

### 2.2. 分离与重连

*   **分离会话 (Detach)**:
    在 Screen 会话中，按下 `Ctrl+a` 组合键，然后松开，再按下 `d` 键。
    ```text
    Ctrl+a d
    ```
    这会将当前 Screen 会话分离，但会话中的程序会继续在后台运行。

*   **列出所有会话**:
    ```bash
    screen -ls
    ```
    输出示例：
    ```
    There are screens on:
            12345.my_session        (Detached)
            67890.pts-0.hostname    (Attached)
    2 Sockets in /run/screen/S-user.
    ```

*   **重连到最近分离的会话**:
    ```bash
    screen -r
    ```

*   **重连到指定会话**:
    使用会话 ID 或会话名称。
    ```bash
    screen -r <session_id>
    ```
    ```bash
    screen -r <session_name>
    ```
    例如：`screen -r 12345` 或 `screen -r my_session`

### 2.3. 彻底关闭会话

*   **在会话内部关闭**:
    在 Screen 会话中，直接退出所有运行的程序，或者输入 `exit` 命令。
    ```bash
    exit
    ```
    或者使用快捷键 `Ctrl+d`。

*   **从外部终止指定会话**:
    使用 `kill` 或 `quit` 命令。
    ```bash
    screen -X -S <session_id> kill
    ```
    ```bash
    screen -X -S <session_name> kill
    ```
    ```bash
    screen -X -S <session_id> quit
    ```
    ```bash
    screen -X -S <session_name> quit
    ```
    `kill` 和 `quit` 在此场景下功能类似，都用于强制终止会话。

## 3. 高级操作

### 3.1. 多窗口管理

Screen 会话可以包含多个窗口，每个窗口运行一个独立的 shell 或程序。

*   **创建新窗口**:
    在 Screen 会话中，按下 `Ctrl+a` 组合键，然后松开，再按下 `c` 键。
    ```text
    Ctrl+a c
    ```
*   **切换窗口**:
    *   下一个窗口: `Ctrl+a n`
    *   上一个窗口: `Ctrl+a p`
    *   切换到指定编号的窗口: `Ctrl+a <number>` (例如 `Ctrl+a 0` 切换到第一个窗口)
*   **显示所有窗口列表**:
    ```text
    Ctrl+a "
    ```
    这会显示一个窗口列表，您可以通过方向键选择并回车切换。

### 3.2. 分屏操作

Screen 支持在单个终端窗口中进行水平或垂直分屏。

*   **水平分屏**:
    ```text
    Ctrl+a S
    ```
    (大写 S)
*   **垂直分屏**:
    ```text
    Ctrl+a |
    ```
*   **在分屏之间切换**:
    ```text
    Ctrl+a Tab
    ```
*   **关闭当前分屏**:
    ```text
    Ctrl+a X
    ```

### 3.3. 会话共享

允许多个用户或从多个终端连接到同一个 Screen 会话，适合协作。

*   **共享会话**:
    ```bash
    screen -x <session_name>
    ```
    所有连接到此会话的用户将看到相同的屏幕内容并可以进行交互。

### 3.4. 锁定会话

锁定当前 Screen 会话，需要输入密码才能解锁，增强安全性。

*   **锁定当前会话**:
    ```text
    Ctrl+a x
    ```
    (小写 x)

### 3.5. 滚动查看历史输出

当输出内容超出屏幕时，可以进入滚动模式查看历史记录。

*   **进入滚动模式**:
    ```text
    Ctrl+a [
    ```
    进入后，可以使用方向键、Page Up/Page Down 键进行滚动。
*   **退出滚动模式**:
    按下 `Esc` 键。

### 3.6. 会话重命名

为当前窗口重命名，方便识别。

*   **重命名当前窗口**:
    ```text
    Ctrl+a A
    ```
    (大写 A)
    输入新名称后按回车。

### 3.7. 会话日志记录

将 Screen 会话的所有输出记录到一个文件中。

*   **开始/停止记录**:
    ```text
    Ctrl+a H
    ```
    (大写 H)
    第一次按下开始记录，再次按下停止记录。日志文件通常保存在当前目录，文件名为 `screenlog.<window_number>`。

## 4. 实用示例

### 4.1. 运行长时间任务并分离

1.  **创建命名会话**:
    ```bash
    screen -S long-task
    ```
2.  **在会话中运行任务**:
    ```bash
    # 例如，运行一个 Python 训练脚本
    python train.py
    ```
3.  **分离会话**:
    按下 `Ctrl+a d`。
4.  **查看所有运行中的会话**:
    ```bash
    screen -ls
    ```
5.  **重新连接到 `long-task` 会话**:
    ```bash
    screen -r long-task
    ```
6.  **强制重连 (如果会话显示为 "Attached" 但实际已断开)**:
    ```bash
    screen -d -r long-task
    ```

### 4.2. 多人协作共享会话

1.  **用户 A 创建会话**:
    ```bash
    screen -S shared-work
    ```
2.  **用户 B 连接到同一会话**:
    ```bash
    screen -x shared-work
    ```
    现在用户 A 和用户 B 将看到相同的终端内容并可以共同操作。

## 5. 常见问题解决

### 5.1. 会话状态显示为 "Attached" 但无法连接

**问题描述**: `screen -ls` 显示会话为 `Attached`，但尝试 `screen -r <session_id>` 时提示无法连接。这通常发生在终端意外关闭或连接中断后。

**解决方案**:
1.  **强制分离会话**:
    ```bash
    screen -d <session_id>
    ```
    例如：`screen -d 12345`
2.  **重新连接会话**:
    ```bash
    screen -r <session_id>
    ```
    例如：`screen -r 12345`

### 5.2. 清理死掉的会话 (Dead Sessions)

**问题描述**: 有些 Screen 会话可能因为异常终止而变成 `Dead` 状态，占用资源。

**解决方案**:
*   **自动清理所有死掉的会话**:
    ```bash
    screen -wipe
    ```
    此命令会列出并移除所有处于 `Dead` 状态的会话。
