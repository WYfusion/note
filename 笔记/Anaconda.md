# Anaconda

### 更新conda到最新版本

- 在开始升级Anaconda之前，首先需要确保conda自身是最新版本。这可以通过运行以下命令来完成：

  ```bash
  conda update conda
  ```

  此命令会检查conda是否有新版本，并提示你进行更新。这是非常重要的步骤，因为更新conda有助于确保后续Anaconda的更新过程更加顺利。

### 更新Anaconda

- 一旦conda更新到最新版本，接下来就可以更新Anaconda了。虽然

  ```bash
  conda update anaconda
  ```

  命令在某些情况下可以直接使用，但更推荐的方法是使用以下命令，因为这个命令会更新所有已安装的包，包括Anaconda本身。

  ```bash
  conda update --all
  ```

  这个命令会检查所有已安装的包，包括Anaconda，并尝试将它们更新到最新版本。请注意，这个过程中可能会遇到一些依赖性问题或版本不兼容的情况，conda会尝试自动解决这些问题，但有时可能需要你手动介入。

### 创建环境

使用以下语句

```bash
conda create --name pytorch python=3.8
```

使用以下语句激活

```bash
conda activate pytorch
```

### 搭建环境

```bash
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```

```bash
conda install pytorch=2.0 torchvision torchaudio cudatoolkit=11.8 -c pytorch
```



### 批量安装软件包

```bash
pip install -r .\requirements.txt
```

.\requirements.txt是当前目录下的requirements.txt文件，该文件里面包含了要求的软件包（一列）

### 删除指定的环境

```bash
 # 列出所有环境（可选）
conda env list
 # 删除名为 "myenv" 的环境
conda remove --name myenv --all
```

### **导出环境为 YAML 文件**：

使用 `conda env export` 命令将当前激活的环境导出为一个 YAML 格式的文件。例如，输入 `conda env export > environment.yml`，这将创建一个名为 `environment.yml` 的文件，其中包含了环境的所有依赖包及其版本信息。

```bash
conda env export > environment.yml
```



### **在目标主机上创建环境**：

将 `environment.yml` 文件复制到目标主机上，然后在目标主机的 Conda 终端中，使用 `conda env create -f environment.yml` 命令来根据 YAML 文件创建环境。这将重建与源主机上相同的环境。

```bash
conda env create -f environment.yml
```

