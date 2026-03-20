在虚拟环境中执行，也可以使用vscode新建`.ipynb`文件

如果文件路径不在C盘，可以用以下语句来打开

```bash
jupyter notebook --notebook-dir=E:
```

在anaconda中建立的虚拟环境中创建Jupyter需要使用以下命令

```bash
conda install nb_conda
```

安装失败的可以试试这个命令

```bash
conda install jupyter notebook
or
pip install notebook
```