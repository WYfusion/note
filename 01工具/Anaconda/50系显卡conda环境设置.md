## 首先是cuda，安装版本12.8
```bash
conda install cuda-toolkit=12.8 cuda-nvcc=12.8 cuda-compiler=12.8 -c nvidia
```

安装完成后输入指令`nvcc --version`出现以下结果表示安装成功
```bash
(wy312) rtx5090@rtx5090:~/data/wy/voxaboxen-main$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Fri_Feb_21_20:23:50_PST_2025
Cuda compilation tools, release 12.8, V12.8.93
Build cuda_12.8.r12.8/compiler.35583870_0
(wy312) rtx5090@rtx5090:~/data/wy/voxaboxen-main$ 
```

## 再安装cudnn
```bash
conda install cudnn -c nvidia
```

## 最后直接安装pytorch就行
```bash
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

主要需要 CUDA Toolkit 支持sm_120