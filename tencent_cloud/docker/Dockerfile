# [推荐] 使用 NVIDIA 的 PyTorch 镜像作为基础镜像，以兼容较新的开源库和 GPU 卡类型
FROM nvcr.io/nvidia/pytorch:23.07-py3

# [推荐] 修改软件源（若在腾讯云使用推荐用内网源）
# [腾讯外网软件源] mirrors.tencent.com
# [腾讯云内网软件源] mirrors.tencentyun.com
ENV TENCENT_MIRRORS="mirrors.tencent.com"
RUN sed -i "s/archive.ubuntu.com/${TENCENT_MIRRORS}/g" /etc/apt/sources.list && \
    sed -i "s/security.ubuntu.com/${TENCENT_MIRRORS}/g" /etc/apt/sources.list && \
    pip config set global.index-url http://${TENCENT_MIRRORS}/pypi/simple && \
    pip config set global.no-cache-dir true && \
    pip config set global.trusted-host ${TENCENT_MIRRORS}

# [推荐] 若使用 NVIDIA 的 PyTorch 镜像，推荐删除默认的 NVIDIA 源，以加快 pip 包查询和安装速度
RUN rm /etc/xdg/pip/pip.conf /etc/pip.conf /root/.pip/pip.conf /root/.config/pip/pip.conf && pip config unset global.extra-index-url

# [基本镜像规范] 安装 openssh-server,  notebook的ssh登录功能以及任务式建模均依赖openssh-server 组件
RUN apt-get update && apt-get install -y openssh-server && apt-get clean && mkdir -p /var/run/sshd

# [Notebook 镜像规范] 配置 /opt/dl/run 启动入口
RUN mkdir -p /opt/dl && echo "cd /home/tione/notebook && jupyter lab --allow-root --no-browser --ip=0.0.0.0 --port=8888 --notebook-dir=/home/tione/notebook --NotebookApp.allow_origin='*' --NotebookApp.token=''" > /opt/dl/run && chmod a+x /opt/dl/run

# [推荐] 使用tini作为entrypoint，方便回收僵尸进程
RUN apt-get update && apt-get install -y tini && apt-get clean
ENTRYPOINT ["/usr/bin/tini", "-g", "--"]

# [可选] 安装 Tikit (不含大数据组件)
RUN pip install tencentcloud-sdk-python==3.0.955 coscmd==1.8.6.31 && \
    pip install --no-dependencies -U tikit

# [自定义] 安装需要的依赖库
RUN pip3 install accelerate bitsandbytes datasets deepspeed evaluate peft protobuf scipy sentencepiece transformers tensorboard wandb