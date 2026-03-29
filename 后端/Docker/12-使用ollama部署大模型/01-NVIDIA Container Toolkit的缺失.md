如若出现：
```bash
rtx5090@rtx5090:~$ docker run -d --gpus=all -v ollama:/root/.ollama -v /home/rtx5090/data/wy:/root/wy -p 11434:11434 -p 2222:22 --name ollama ollama/ollama

Unable to find image 'ollama/ollama:latest' locally
latest: Pulling from ollama/ollama
5d7637d33c73: Pull complete 
1b83b22fdc34: Pull complete
20043066d3d5: Pull complete 
2ebb8c780efd: Pull complete 
83cbce76a497: Download complete 
Digest: sha256:2c9595c555fd70a28363489ac03bd5bf9e7c5bdf2890373c3a830ffd7252ce6d
Status: Downloaded newer image for ollama/ollama:latest
28c077de19673870208bfcc70c242ad4df66b93f3455a25c126d7f8e7e4c95c1
docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]

Run 'docker run --help' for more information
```
的报错，这意味着 **Docker 根本不知道如何调用你的显卡**。这是因为你缺少了 **NVIDIA Container Toolkit**（Docker 和显卡驱动之间的“翻译官”）。

### 解决方案（只需执行一次）

请依次执行以下 3 步命令来安装并配置该工具：
#### 第一步：添加 NVIDIA 容器工具包的源
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

#### 第二步：安装工具包
```bash
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

#### 第三步：配置 Docker 运行时并重启 Docker（关键）
这一步会自动修改 `/etc/docker/daemon.json` 文件，让 Docker 能够识别 NVIDIA 运行时。
```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```