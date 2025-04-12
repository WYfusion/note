终端使用以下语句：

```bash
tensorboard --logdir=tensorboard/logs/runs/test/1							# logdir=事件文件所在文件夹名
```

调整服务器端口号

```bash
tensorboard --logdir=tensorboard/logs/runs/test/1 --port=6007				# 调整服务器号为6007
```

注意后面的路径要和命令行所在的根路径相接才会正常读取到数据的。