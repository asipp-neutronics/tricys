FROM openmodelica/openmodelica:v1.24.5-ompython

# 1. 先以 root 身份安装所有系统依赖
USER root
RUN apt-get update && apt-get install -y \
    vim \
    curl \
    git \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# 2. 将应用程序代码复制到容器中
WORKDIR /tricys
COPY . /tricys

# 3. 安装 Python 依赖
#RUN pip install --no-cache-dir -r requirements.txt

# 2025-07-24 08:15:27,106 - OMPython - ERROR - OMC Server did not start. Please start it! Log-file says:
# Error: You are trying to run OpenModelica as a server using the root user.
# This is a very bad idea:
# * The socket interface does not authenticate the user.
# * OpenModelica allows execution of arbitrary commands.
# Execution failed!

# 4. 创建一个新的、有特权的普通用户
RUN useradd --create-home --shell /bin/bash appuser && \
    usermod -aG sudo appuser && \
   echo "appuser ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# 5. 更改 /app 目录及其所有文件的所有权，使其属于新创建的 appuser
RUN chown -R appuser:appuser /tricys

# 6. 切换到新创建的普通用户
USER appuser

# 7. 更新pip
RUN pip install --upgrade pip setuptools wheel

# 8. 安装项目依赖
RUN make dev-install

# 9. 设置环境变量
ENV PATH="/home/appuser/.local/bin:${PATH}"
