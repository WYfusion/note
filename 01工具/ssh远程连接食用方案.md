需要在VSCode中安装 **Remote - SSH** 扩展

主要参考：[Windows安装和启动SSH服务_windows 安装ssh-CSDN博客](https://blog.csdn.net/qq_33594636/article/details/128849482)
被控制端需要安装OpenSSH.Server，最好把客户端和服务端都安装好。
控制端安装客户端，默认已经安装。

#### 安装方法：
1. 系统设置-主页-搜索添加可选功能-添加功能-搜索OpenSSH，选择安装。
![[Pasted image 20250320141955.png|600]]
需确保系统为**Windows 10 1809+ 或 Windows Server 2019+**，否则需使用方法2。
	验证：
```powershell
	Get-WindowsCapability -Online | Where-Object Name -like 'OpenSSH*'
```
1. 也可以使用git-hub中下载：选择[OpenSSH-Win64.zip](https://github.com/PowerShell/Win32-OpenSSH/releases/download/v9.8.1.0p1-Preview/OpenSSH-Win64.zip)，下载解压到C:\Program Files\OpenSSH-Win64。注意添加环境变量到系统path中去。`C:\Program Files\OpenSSH-Win64`
再于C:\Program Files\OpenSSH-Win64内执行。最好使用powershell管理员
```powershell
cd "C:\Program Files\OpenSSH-Win64"
powershell.exe -ExecutionPolicy Bypass -File install-sshd.ps1
```
#### 基本配置
若需在目标主机上启用多端口 SSH 服务：

1. 修改 `/etc/ssh/sshd_config`，添加 `Port` 行：
```shell
Port 22     # 保留默认端口 
Port 2222   # 新增端口
```
### 配置防火墙（开放 SSH 端口）

#### 1. 开放默认端口（22）
```powershell
# 允许入站 TCP 流量到 22 端口 
New-NetFirewallRule -Name "OpenSSH-Server" -DisplayName "OpenSSH Server (sshd)" -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22
```

- 验证规则是否生效：
```shell
Get-NetFirewallRule -Name "OpenSSH-Server"
```

#### 2. 开放自定义端口（如 2222）

若需修改 SSH 端口（例如改为 2222）：

1. **修改 SSH 配置文件**： 编辑 `C:\ProgramData\ssh\sshd_config`，添加或修改行：`Port 2222`。
2. **配置防火墙规则**：
```Powershell
New-NetFirewallRule -Name "SSH-AltPort" -DisplayName "SSH Alternate Port" -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 2222
```


3. **检查端口监听**：
```Powershell
#查看指定端口是否处于监听状态 
Get-NetTCPConnection -LocalPort 22 -State Listen
```
- 若端口未显示，可能服务未正确启动或防火墙规则未生效
    
4. **重启 SSH 服务**：
```Powershell
Restart-Service sshd
```
使sshd服务开机自启动
```powershell
C:\Windows\System32\sc.exe config sshd start= auto
```

开启sshd服务
```powershell
net start sshd
```


可通过防火墙高级设置界面快速定位端口：
1. 打开 **控制面板 → 系统和安全 → Windows Defender防火墙 → 高级设置**
2. 在 **入站规则** 或 **出站规则** 列表中，按端口排序或搜索
#### 有密、有码
连接设置简单
```powershell
ssh 用户名@本机IP   # 例子ssh Administrator@10.18.11.197
```
回车，接着输入用户密码即可。



#### 免密、无码
连接设置稍难
### 1. **生成SSH密钥对（如果尚未生成）**

最好使用powershell管理员在本地计算机上生成密钥对：
```powershell
ssh-keygen -t rsa -b 4096 -C "your_comment"
```
（默认保存路径：`C:\Users\YourUser\.ssh\id_rsa`（私钥）和 `id_rsa.pub`（公钥）。
生成时可选设置密码短语，若需完全无密码，直接回车跳过。）
4096不必管，是种加密方法。双引号内容是你的密钥标识

---
### 2. **将公钥复制到目标主机**

需将本地公钥 `id_rsa.pub` 内容添加到目标主机的 指定 用户的 `~/.ssh/authorized_keys` 文件中(注意：authorized_keys文件内的公钥必须没有换行，**若多个用户免密则按行写入公钥**)。

#### 内容验证：
- 目标主机:
```powershell
cat ~/.ssh/authorized_keys  # Linux
type C:\Users\dell\.ssh\authorized_keys  # Windows
```

#### 权限验证：
- 目录权限：    
```powershell
chmod 700 ~/.ssh            # Linux
icacls "C:\Users\dell\.ssh" /inheritance:r /grant:r "dell:(F)"  # Windows
```
- 文件权限：
```
chmod 600 ~/.ssh/authorized_keys  # Linux
icacls "C:\Users\dell\.ssh\authorized_keys" /inheritance:r /grant:r "dell:(R)"  # Windows
```


### 3. **检查目标主机的SSH服务器配置**
确保目标主机的 `/etc/ssh/sshd_config` 文件(注意无后缀文件类型)内包含以下配置：
```bash
PubkeyAuthentication yes
AuthorizedKeysFile .ssh/authorized_keys
StrictModes no  # 允许密钥文件权限宽松（Windows特有）
PasswordAuthentication no  # 关键
```
这里可以设置多行Port端口号，建议选择高位端口（如 `50000~65535`）以提高安全性。

### 4. **重启SSH服务使配置生效**
```powershell
sudo systemctl restart sshd  # Linux系统
Restart-Service sshd        # Windows系统（若目标主机为Windows）
```

**本地回环测试** 在目标主机上执行本地 SSH 连接测试：  
```Powershell
ssh -v -i C:\Users\dell\.ssh\id_rsa dell@localhost
```
- 若成功，说明服务端配置正确，问题在外部网络或客户端；
- 若失败，检查 `C:\ProgramData\ssh\logs\sshd.log` 日志。

### 连接
```powershell
ssh -vvv -i "C:\Users\31530\.ssh\id_ed25519" dell@172.22.208.130
```
可以在VScode中键入`Ctrl+Shift+P`，再键入
```shell
Remote-SSH: Add New SSH Host
```
将上面连接命令添加到`C:\Users\你的用户名\.ssh\config`中即可。
设置端口，注意和目标主机的开放端口一致
```powershell
ssh -vvv -p 2222 -i "C:\Users\31530\.ssh\id_rsa" dell@172.22.208.130
```

示例：
```shell
Host A5000
  HostName 172.22.208.130
  User dell
  Port 22
  IdentityFile ~/.ssh/id_rsa
```