WSL是**Windows 提供的一种工具**，允许用户在 Windows 上运行原生的 Linux 二进制程序（如 Bash 脚本、GNU 工具、Linux 命令行工具等），而无需安装双系统或虚拟机。同时VMware无法使用本机显卡，但是使用WSL可以直接使用显卡。不过WSL是纯命令行，也有一些GUI的组件可用，但不推荐。

[VScode](https://code.visualstudio.com/docs/remote/wsl)



前提条件
windows机器需要支持虚拟化，并且需要在BIOS中开启虚拟化技术，因为WSL2基于hyper-V。
查看是否开启虚拟化
按住Windows+R输入cmd打开命令行，输入

```
systeminfo
```

可以看到如下字样，代表电脑已经支持虚拟化，可继续安装

```bash
Hyper-V 要求:     虚拟机监视器模式扩展: 是
                  固件中已启用虚拟化: 是
                  二级地址转换: 是
                  数据执行保护可用: 是
```



 将电脑开发者模式打开

#### 开启“适用于Linux的Windows子系统”

找到`控制面板`-`程序和功能`-`启用或关闭Windows功能`，选中“适用于Linux的Windows子系统”，然后点击`确定`

再重启电脑。



#### 3.安装 WSL 2 之前，必须启用“虚拟机平台”可选功能。 计算机需要虚拟化功能才能使用此功能。
以管理员身份打开PowerShell并运行：

```shell
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
```

```shell
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
```

**安装 WSL**
 在 PowerShell 中运行以下命令以安装 WSL：

```bash
wsl --install
```

这将自动安装最新版本的 WSL 和一个默认的 Linux 发行版（例如 Ubuntu）。

也可以查询可以安装的Ubuntu版本

```powershell
wsl --list --online
```

```powershell
wsl --install -d Ubuntu-22.04
```


**重启电脑**
 安装完成后，重启电脑以完成配置。

打开 `PowerShell`，然后在安装新的 Linux 发行版时运行以下命令，将 WSL 2 设置为默认版本：

```shell
wsl --set-default-version 2
```

**进入 WSL 环境**
 打开命令行，进入 WSL：

```bash
wsl
```

**导航到项目目录**
 在 WSL 中，导航到位于 Windows 中的项目路径。注意，wsl的导航需要用：`/mnt/`来指引

```bash
cd /mnt/e/202409_202508/code/
```

默认安装的是 Ubuntu

#### 在 WSL 中安装 Miniconda 
如果你尚未安装 `conda`，可以通过以下步骤在 WSL 中安装 Miniconda（推荐）。
1. **下载 Miniconda 安装脚本**：在 WSL 中运行以下命令下载 Miniconda 安装脚本：

   ```bash
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   ```

2. **运行安装脚本**：
    执行以下命令安装 Miniconda：

   ```bash
   bash Miniconda3-latest-Linux-x86_64.sh
   ```

   按照提示完成安装，推荐选择默认路径（如 `~/miniconda3`）。

3. **激活 conda 环境**：
    安装完成后，运行以下命令激活 `conda`：

   ```bash
   source ~/miniconda3/etc/profile.d/conda.sh
   ```

   或者将其添加到你的 `~/.bashrc` 中，以便每次启动 WSL 时自动加载：

   ```bash
   echo "source ~/miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc
   source ~/.bashrc
   ```

4. **验证安装**：
    验证 `conda` 是否安装成功：

   ```bash
   conda --version
   ```

   示例输出:

   ```bash
   conda 4.12.0
   ```

#### **检查 `conda` 是否已安装**

如果你已经在 WSL 中安装了 `conda`，但 `conda` 命令不可用，可能是环境变量未正确配置。

1. **检查 `conda` 的路径**：
    查找 `conda` 的安装路径：

   ```bash
   find ~/ -name "conda" -type f
   ```

   如果输出类似以下路径：

   ```bash
   /home/username/miniconda3/bin/conda
   ```

   表示你已安装 `conda`，但未添加到 `PATH`。

2. **手动添加路径到 `PATH`**：
    将 `conda` 的路径添加到环境变量中。例如：

   ```bash
   export PATH="/home/username/miniconda3/bin:$PATH"
   ```

   如果希望永久生效，可以将上述命令追加到 `~/.bashrc` 文件中：

   ```bash
   echo 'export PATH="/home/username/miniconda3/bin:$PATH"' >> ~/.bashrc
   source ~/.bashrc
   ```

3. **验证 `conda` 是否可用**：
    再次运行以下命令检查

   ```bash
   conda --version
   ```