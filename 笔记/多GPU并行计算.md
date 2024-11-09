# 多GPU并行计算

torch.cuda.device_count() 来获取系统中可用的GPU数量。

```python
# GPU check
logging.info(f'GPU is available: {torch.cuda.is_available()}')
if torch.cuda.is available():
	gpu num= torch.cuda.device count()
    logging.info(f"Train model on {gpu num} GPus:")
    for i in range(gpu_num):
		print('\t GPu {}.:{}'.format(i,torch.cuda.get device name(i)))
```

### 使用DataParallel

这是最简单的方法，可以自动将数据分割并发送到多个GPU上，然后再汇总结果。只需将模型包裹在 nn.DataParallel 中即可。例如：

```python
model = nn.Linear(10, 5)
model = nn.DataParallel(model)
model.to(device)
```

夜雨飘零项目中：

#### 使用一块GPU

`CUDA_VISIBLE_DEVICES=0 python train.py --configs=configs/bi_lstm.yml`

```bash
$env:CUDA_VISIBLE_DEVICES = "0"
python train.py --configs=configs/bi_lstm.yml
```

或者在Power Shell中调用cmd.exe

```bash
cmd /c "CUDA_VISIBLE_DEVICES=0 python train.py --configs=configs/bi_lstm.yml"
```

#### 使用此两块GPU

`CUDA_VISIBLE_DEVICES = "0,1" torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py --configs=configs/bi_lstm.yml`

```bash
$env:CUDA_VISIBLE_DEVICES = "0,1"
# 假设 torchrun 可在当前 PowerShell 会话中直接调用
torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py --configs=configs/bi_lstm.yml
```

