在局域网内将一台 Windows 主机的文件夹挂载到另一台 Windows 主机上。

# 源主机端
1. **创建共享文件夹**  
    右键点击需要共享的文件夹，选择「属性」→「共享」选项卡→点击「共享...」按钮。在弹出窗口中选择共享用户（如`Everyone`），设置权限级别（读取或读写），点击「共享」完成基础设置。

2. **高级共享配置（可选）**  
    若需更精细控制，返回「共享」选项卡→点击「高级共享」→勾选「共享此文件夹」并设置共享名称。点击「权限」可配置特定用户或组的访问权限，例如仅允许某用户读写。

3. **启用 SMB 协议**  
    若共享访问失败，需确认 SMB 服务已启用：
    - 打开「控制面板」→「程序」→「启用或关闭 Windows 功能」→勾选「SMB 1.0/CIFS 文件共享支持」并重启电脑

# 本地主机
同Ubuntu挂载[[Ubuntu文件夹挂载到Windows#^6abc70|Windows端一致]]。