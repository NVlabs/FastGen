# Base image 25.11-py3 ships torch 2.10.0a0 / CUDA 13.0 / Python 3.12
FROM nvcr.io/nvidia/pytorch:25.11-py3

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
        S5CMD_SHA256="f64dcaabbb837ce223d716c9e64b4829e5cd780f151a42f2d2cb7edccedb7b06"; \
    else \
        S5CMD_ARCH="Linux-64bit"; \
        S5CMD_SHA256="de5efa37941ef83e892662aefc936603a1a8acbc5a6a9e003a71721a7140b610"; \
    fi && \
    S5CMD_ARCHIVE="s5cmd_${S5CMD_VERSION}_${S5CMD_ARCH}.tar.gz" && \
    wget "https://github.com/peak/s5cmd/releases/download/v${S5CMD_VERSION}/${S5CMD_ARCHIVE}" && \
    echo "${S5CMD_SHA256}  ${S5CMD_ARCHIVE}" | sha256sum --check --strict - && \
    tar -xf "${S5CMD_ARCHIVE}" && \
    install s5cmd /usr/local/bin/ && \
    rm s5cmd "${S5CMD_ARCHIVE}"

# Set build optimization flags
ENV MAX_JOBS=4
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0;10.0;12.0f+PTX"

# Upgrade pip
RUN pip install --no-cache-dir --root-user-action=ignore --upgrade pip setuptools wheel
# numpy<2.0.0: required for the preinstalled numba and thinc
# Pin the PEFT/TorchAO pair tested with diffusers 0.40 and torch 2.10.
RUN pip install --no-cache-dir --root-user-action=ignore wandb[media] \
    diffusers==0.40.0 \
    transformers==5.15.0 \
    peft==0.20.0 \
    torchao==0.16.0 \
    accelerate \
    safetensors \
    scipy \
    attrs \
    matplotlib \
    psutil \
    einops \
    hydra-core \
    pyyaml \
    "imageio[ffmpeg]" \
    tqdm \
    click \
    webdataset \
    av \
    loguru \
    boto3 \
    timm \
    ftfy \
    opencv-python-headless \
    sentencepiece \
    openai \
    python-dotenv \
    "numpy<2.0.0"
RUN pip uninstall -y apex
RUN pip uninstall -y pynvml && pip install --no-cache-dir --root-user-action=ignore nvidia-ml-py

WORKDIR /workspace

# Entry point; without a command, the container opens a shell
RUN printf '%s\n' '#!/bin/bash' 'exec "$@"' > /entry.sh && chmod a+x /entry.sh
ENTRYPOINT ["/entry.sh"]
CMD ["/bin/bash"]
