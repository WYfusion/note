# Screen 命令详细指南

## 安装 Screen
```shell
sudo apt install screen    # Debian/Ubuntu
```

```shell
sudo yum install screen    # CentOS/RHEL
```

## 基本操作
### 新建会话
```shell
screen
```
创建一个带名称的 screen 会话
```shell
screen -S 会话名称
```

### 分离与重连

从当前会话分离（暂时离开）,是先按下`Ctrl+a`再按`d`
```shell
Ctrl+a d
```
重新连接到最近的一个会话
```shell
screen -r
```
重新连接到指定会话 (使用会话ID或名称)
```shell
screen -r 会话ID
```

```shell
screen -r 会话名称
```
### 列出会话
查看所有会话列表
```shell
screen -ls
```
### 彻底关闭会话
在会话中直接退出并终止该会话
```shell
exit
```
或者使用快捷键
```shell
Ctrl+d
```

从外部终止指定会话
```shell
screen -X -S 会话ID kill
```
```shell
screen -X -S 会话名称 kill
```
```shell
screen -X -S 会话ID quit
```
```shell
screen -X -S 会话名称 quit
```

## 高级操作
### 多窗口管理
在当前会话中创建新窗口
```shell
Ctrl+a c
```

在窗口之间切换
```shell
Ctrl+a n    # 下一个窗口
```

```shell
Ctrl+a p    # 上一个窗口
```

```shell
Ctrl+a 数字  # 切换到指定编号的窗口
```

显示所有窗口列表

```shell
Ctrl+a "
```

### 分屏操作
水平分屏
```shell
Ctrl+a S
```

垂直分屏
```shell
Ctrl+a |
```

在分屏之间切换

```shell
Ctrl+a Tab
```

关闭当前分屏

```shell
Ctrl+a X
```

### 会话共享
允许其他用户连接到您的会话(适合协作)
```shell
screen -x 会话名称
```

### 锁定会话
锁定当前会话(需要密码解锁)
```shell
Ctrl+a x
```

### 滚动查看历史输出
进入滚动模式
```shell
Ctrl+a [
```
使用方向键或Page Up/Down滚动
按ESC退出滚动模式

### 会话的重命名
```shell
Ctrl+a A
```
### 会话的日志记录
开始记录当前会话到文件
```shell
Ctrl+a H
```

## 实用示例
```shell
创建一个名为"long-task"的会话并在其中运行长时间任务
screen -S long-task
# 运行您的命令，例如 python train.py
# 然后按 Ctrl+a d 分离会话
# 查看所有运行中的会话
screen -ls
# 重新连接到long-task会话
screen -r long-task
如果会话出现 "Attached" 状态，可以强制重连
screen -d -r long-task
在不同机器上共享一个会话（协作场景）
screen -x 会话名称
```

## 常见问题解决
会话状态显示为 "Attached" 但实际已断开连接
```shell
screen -d 会话ID    # 先强制分离
```

```shell
screen -r 会话ID    # 然后重连
```
清理死掉的会话
```shell
screen -wipe
```
