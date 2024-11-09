# $Bash(Bourne\ Again\ Shell)$

是一种流行的命令行解释器，用于与操作系统进行交互。以下是 Bash 语法的详细介绍，包括基本命令、变量、控制结构、函数等。

### 1. 基本命令

#### 1.1 文件和目录操作

- **列出目录内容**:

  ```bash
  ls         # 列出当前目录的文件和子目录
  ls -l      # 以长格式列出
  ls -a      # 显示所有文件，包括隐藏文件
  ```

- **更改目录**:

  ```bash
  cd ./path/to/directory  # 进入指定目录，要求当前已在该根目录下
  cd ..                   # 返回上级目录
  cd ~                    # 进入用户主目录
  D:						# 进入D盘符 
  ```

- **创建和删除目录**:

  ```bash
  mkdir new_directory     # 创建新目录
  rmdir empty_directory    # 删除空目录
  rm -r directory_name     # 递归删除目录及其内容
  ```

#### 1.2 文件操作

- 复制、移动和删除文件

  ```bash
  cp source.txt destination.txt   # 复制文件
  mv oldname.txt newname.txt      # 移动或重命名文件
  rm file.txt                     # 删除文件
  ```

### 2. 变量

#### 2.1 定义变量

```bash
variable_name="value"  # 定义变量，注意等号两边不能有空格
```

#### 2.2 使用变量

```bash
echo $variable_name   # 输出变量值
```

#### 2.3 读取用户输入

```bash
read variable_name   # 从标准输入读取值并存入变量
```

### 3. 控制结构

#### 3.1 条件语句

- **if 语句**:

  ```bash
  if [ condition ]; then
      # 代码块
  elif [ another_condition ]; then
      # 代码块
  else
      # 代码块
  fi
  ```

- **示例**:

  ```bash
  if [ -f "file.txt" ]; then
      echo "file.txt exists."
  else
      echo "file.txt does not exist."
  fi
  ```

#### 3.2 循环语句

- **for 循环**:

  ```bash
  for variable in list; do
      # 代码块
  done
  ```

- **示例**:

  ```bash
  for i in {1..5}; do
      echo "Iteration $i"
  done
  ```

- **while 循环**:

  ```bash
  while [ condition ]; do
      # 代码块
  done
  ```

- **示例**:

  ```bash
  count=1
  while [ $count -le 5 ]; do
      echo "Count is $count"
      ((count++))  # 自增
  done
  ```

### 4. 函数

#### 4.1 定义函数

```bash
function_name() {
    # 代码块
}
```

#### 4.2 调用函数

```bash
function_name   # 调用函数
```

#### 4.3 示例

```bash
greet() {
    echo "Hello, $1!"  # $1 是传入的第一个参数
}

greet "World"  # 输出: Hello, World!
```

### 5. 管道和重定向

- **管道**: 将一个命令的输出作为另一个命令的输入。

  ```bash
  command1 | command2
  ```

- **重定向输出到文件**:

  ```bash
  command > file.txt     # 将输出重定向到文件，覆盖原文件
  command >> file.txt    # 将输出追加到文件
  ```

- **重定向输入**:

  ```bash
  command < file.txt
  ```

### 6. 脚本文件

- **创建脚本文件**:

  1. 使用文本编辑器创建文件，如 `script.sh`。

  2. 在文件的第一行添加 Shebang:

     ```bash
     #!/bin/bash
     ```

  3. 添加脚本内容。

- **赋予执行权限**:

  ```bash
  chmod +x script.sh
  ```

- **运行脚本**:

  ```bash
  ./script.sh
  ```

### 7. 常用命令

- **查看当前工作目录**:

  ```bash
  pwd
  ```

- **显示文件内容**:

  ```bash
  cat file.txt         # 显示文件内容
  less file.txt        # 分页显示文件内容
  head file.txt        # 显示文件开头部分
  tail file.txt        # 显示文件末尾部分
  ```

- **查找文件**:

  ```bash
  find /path -name "filename"  # 查找文件
  ```

- **搜索文本**:

  ```bash
  grep "pattern" file.txt      # 在文件中搜索模式
  ```

### 8. 小技巧

- **使用 `tab` 键自动补全**。
- **使用 `Ctrl + R` 进行历史命令搜索**。