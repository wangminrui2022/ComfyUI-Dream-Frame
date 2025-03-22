#!/bin/bash

# 定义 Python 可执行文件的路径
PYTHON_PATH="./venv/bin/python3"

# 定义主程序的路径
MAIN_PY_PATH="main.py"

# 运行命令
$PYTHON_PATH $MAIN_PY_PATH --preview-method auto --listen 0.0.0.0 --port 48569

