FROM nvcr.io/nvidia/pytorch:25.06-py3

# Defines the architecture (amd64 or arm64) automatically during build
ARG TARGETARCH

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Set the timezone
ENV TZ=America/Los_Angeles
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install system dependencies
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ninja-build \
        cmake \
        wget \
        ca-certificates \
        git \
        curl \
        unzip \
        python3-tk \
    && rm -rf /var/lib/apt/lists/* \
    && cmake --version \
    && echo "✅ All system dependencies installed successfully"

# Install s5cmd with architecture logic
RUN S5CMD_VERSION="2.1.0-beta.1" && \
    if [ "$TARGETARCH" = "arm64" ]; then \
        S5CMD_ARCH="Linux-arm64"; \
    else \
        S5CMD_ARCH="Linux-64bit"; \
    fi && \
    wget "https://github.com/peak/s5cmd/releases/download/v${S5CMD_VERSION}/s5cmd_${S5CMD_VERSION}_${S5CMD_ARCH}.tar.gz" && \
    tar -xf "s5cmd_${S5CMD_VERSION}_${S5CMD_ARCH}.tar.gz" && \
    install s5cmd /usr/local/bin/ && \
    rm s5cmd "s5cmd_${S5CMD_VERSION}_${S5CMD_ARCH}.tar.gz"

# Set build optimization flags
ENV MAX_JOBS=4
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0;10.0" 

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel
# numpy<2.0.0 is required for the preinstalled numba and thinc
RUN pip install wandb[media] \
    diffusers==0.35.1 \
    transformers==4.49.0 \
    accelerate \
    safetensors \
    scipy \
    einops \
    jupyter \
    hydra-core \
    imageio \
    tqdm \
    webdataset \
    av \
    loguru \
    boto3 \
    timm \
    ftfy \
    opencv-python-headless \
    sentencepiece \
    "numpy<2.0.0" 
RUN pip uninstall -y apex 

WORKDIR /workspace

# Entry point
RUN (printf '#!/bin/bash\nexec \"$@\"\n' >> /entry.sh) && chmod a+x /entry.sh
ENTRYPOINT ["/entry.sh"]

