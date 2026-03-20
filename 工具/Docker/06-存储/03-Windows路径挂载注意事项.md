# Windows 路径挂载注意事项

## 适用场景
- 在 Windows 上使用 Docker Desktop 开发。
- 遇到 `invalid reference format` 或 `not found` 报错。
- 容器内读写 Windows 文件极慢。

## 1. 路径写法 (PowerShell vs CMD vs WSL)

### 1.1 PowerShell (推荐)
PowerShell 是 Windows 下最常用的终端。
- **当前目录**: 使用 `${PWD}`。
- **引号**: **必须加引号**，防止路径中有空格导致参数错位。
- **换行**: 使用反引号 `` ` ``。

```powershell
docker run -d `
  -v "${PWD}:/app" `
  -v "D:\My Data\Images:/data" `
  nginx
```

### 1.2 CMD (命令提示符)
- **当前目录**: 使用 `%cd%`。
- **换行**: 使用 `^`。

```cmd
docker run -d ^
  -v "%cd%:/app" ^
  nginx
```

### 1.3 WSL2 (Linux 子系统)
如果你在 WSL2 的 Ubuntu 终端里操作：
- Windows 的 C 盘挂载在 `/mnt/c`。
- **注意**: 尽量不要在 WSL2 里挂载 `/mnt/c` 下的文件，**IO 性能极差**（跨文件系统）。
- **推荐**: 把代码放在 WSL2 自己的文件系统里（如 `/home/user/project`）。

## 2. 常见报错与坑点

### 坑点 1：路径中有空格
**报错**: `docker: invalid reference format`
**原因**: `D:\My Data` 中间有空格，Docker 以为 `Data` 是下一个参数。
**解决**: 用双引号把路径包起来：`-v "D:\My Data:/data"`。

### 坑点 2：盘符冒号问题
**报错**: `invalid mode: /data`
**原因**: Windows 路径 `C:\Users` 包含冒号，Docker 的 `-v` 参数也用冒号分隔（`宿主:容器`），导致解析混乱。
**解决**: Docker Desktop 会自动处理盘符，但建议始终使用**绝对路径**并加引号。

### 坑点 3：文件权限 (Permission Denied)
Windows 的文件系统 (NTFS) 没有 Linux 的 `chmod` 权限概念。
- 挂载进容器后，所有文件通常显示为 `root` (777) 或 `1000`。
- **无法在容器内修改 Windows 文件的权限**（`chmod` 无效）。
- 如果数据库容器（如 Postgres）要求配置文件必须是 600 权限，挂载 Windows 文件会导致启动失败。
- **解决**: 使用 Docker Volume 代替 Bind Mount，或者在 WSL2 内部运行。

### 坑点 4：换行符 (CRLF vs LF)
**现象**: 挂载进去的 Shell 脚本 (`run.sh`) 运行报错 `\r: command not found`。
**原因**: Windows 编辑器默认用 `CRLF` (\r\n) 换行，Linux 只认 `LF` (\n)。
**解决**: 在 VS Code 右下角点击 `CRLF` 切换为 `LF`，然后保存。

## 3. 性能警告 (WSL2)
> [!DANGER] 性能杀手
> 在 WSL2 中使用 Docker 时，**千万不要**把代码放在 Windows 盘（`/mnt/c/...`）然后挂载进容器。
> 跨文件系统 IO 非常慢，训练速度可能下降 10 倍。
> **正确做法**: 把代码 `git clone` 到 WSL2 的 `/home/user/` 目录下，然后挂载。

