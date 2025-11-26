# CUDA 11.8 + cuDNN8 Runtime（推荐用于部署推理或训练）
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# 维护者信息
LABEL maintainer="your_name <your@email.com>"
ENV DEBIAN_FRONTEND=noninteractive

# ----------------------------------------------------------
# 基础环境
# ----------------------------------------------------------
RUN sed -i 's@archive.ubuntu.com@mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list && \
    sed -i 's@security.ubuntu.com@mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list

RUN apt-get update && apt-get install -y \
    git wget curl vim \
    python3 python3-pip python3-venv python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# pip 换源（提升 CI/CD 速度）
RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# ----------------------------------------------------------
# 安装 PyTorch 1.13（cu117）
# ----------------------------------------------------------
# PyTorch 1.13 没有 cu118 版本，但 cu117 可在 CUDA11.8 上完全正常运行
RUN pip3 install torch==1.13.1+cu117 \
    torchvision==0.14.1+cu117 \
    torchaudio==0.13.1 \
    --extra-index-url https://download.pytorch.org/whl/cu117

# ----------------------------------------------------------
# 检查 PyTorch CUDA（可选，可删除）
# ----------------------------------------------------------
RUN python3 - <<'EOF'
import torch
print("CUDA Available:", torch.cuda.is_available())
print("Torch CUDA Version:", torch.version.cuda)
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
EOF

WORKDIR /workspace
CMD ["bash"]
