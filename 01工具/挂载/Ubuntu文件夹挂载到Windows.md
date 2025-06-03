一般流程如下
# Ubuntu端
## 安装 Samba 服务
```bash
# 更新软件源
sudo apt update
# 安装 Samba
sudo apt install samba -y
```
也可在Ubuntu内的文件管理器里面 右键指定局域网内的文件夹(例如`dataset`)，选择`本地网络共享`。这同样会让你给主机添加Samba服务。这里也可以赋予共享文件一些权限。

## 为 Samba 配置用户密码
Samba 需单独设置用户密码（Ubuntu 已有系统密码与Samba的不是一个逻辑的，需要另外在主机内`新建/直接用当前已有的`用户）：
```bash
sudo smbpasswd -a your_username  # your_username是已有的用户名
```
然后根据提示输入两次密码即可
#### 创建系统用户(可选)
```bash
sudo useradd -m my_name  # -m 选项：创建用户主目录,便于Samba连接，但也可以直接用原有用户
```

## 重启 Samba 服务使配置生效
```bash
sudo systemctl restart smbd
```
## 查找 Ubuntu 主机 IP
这一步是为了在Windows主机中连接Ubuntu主机所用
在 Ubuntu 终端运行 `ip a` 或 `ifconfig`，查看局域网 IP (例如`192.168.1.100`)


# Windows端

^6abc70

## 访问共享路径
下面二选一
### 图形界面映射网络驱动器
打开「此电脑」→点击「计算机」选项卡→选择「映射网络驱动器」  
    `\\<Ubuntu_IP>\shared_folder`（例如 `\\192.168.1.100\dataset`）。
### 命令行快速挂载
按下`Win+R`输入`cmd`，执行命令：
```bash
net use Z: \\192.168.1.100\dataset /user:用户名 密码 /persistent:yes
```
## 输入认证信息
弹出登录窗口时，输入 Ubuntu 端配置的 Samba 用户名和密码即可
