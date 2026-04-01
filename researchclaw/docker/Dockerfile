# ResearchClaw experiment sandbox — GPU-enabled, isolated environment.
# Build: docker build -t researchclaw/experiment:latest researchclaw/docker/
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# System packages: Python 3.11, build tools, common native deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv python3-pip \
        gcc g++ cmake make gfortran \
        libopenblas-dev liblapack-dev \
        git curl ca-certificates \
        iptables iproute2 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && python3 -m pip install --upgrade pip setuptools wheel \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Core ML/scientific Python stack
RUN python3 -m pip install \
        torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 \
    && python3 -m pip install \
        numpy scipy scikit-learn pandas matplotlib seaborn \
        tqdm gymnasium networkx torchdiffeq PyYAML \
    && python3 -m pip install "gymnasium[mujoco]" "gymnasium[box2d]" mujoco

# Extended ML ecosystem — vision, metrics, utilities
RUN python3 -m pip install \
        timm einops torchmetrics Pillow \
        h5py tensorboard albumentations kornia

# LLM / Transformer training stack
RUN python3 -m pip install \
        transformers>=4.46.0 \
        datasets>=3.0.0 \
        accelerate>=1.0.0 \
        peft>=0.13.0 \
        trl>=0.12.0 \
        bitsandbytes>=0.44.0 \
        sentencepiece protobuf tokenizers safetensors \
        evaluate rouge-score

# Pre-cache standard datasets for offline use
RUN mkdir -p /opt/datasets && \
    python3 -c "import torchvision; torchvision.datasets.CIFAR10(root='/opt/datasets', train=True, download=True); torchvision.datasets.CIFAR10(root='/opt/datasets', train=False, download=True)" && \
    python3 -c "import torchvision; torchvision.datasets.CIFAR100(root='/opt/datasets', train=True, download=True); torchvision.datasets.CIFAR100(root='/opt/datasets', train=False, download=True)" && \
    python3 -c "import torchvision; torchvision.datasets.FashionMNIST(root='/opt/datasets', train=True, download=True); torchvision.datasets.FashionMNIST(root='/opt/datasets', train=False, download=True)" && \
    python3 -c "import torchvision; torchvision.datasets.MNIST(root='/opt/datasets', train=True, download=True); torchvision.datasets.MNIST(root='/opt/datasets', train=False, download=True)" && \
    python3 -c "import torchvision; torchvision.datasets.STL10(root='/opt/datasets', split='train', download=True); torchvision.datasets.STL10(root='/opt/datasets', split='test', download=True)" && \
    python3 -c "import torchvision; torchvision.datasets.SVHN(root='/opt/datasets', split='train', download=True); torchvision.datasets.SVHN(root='/opt/datasets', split='test', download=True)" && \
    chmod -R a+r /opt/datasets

# Non-root user for security
RUN groupadd -r researcher && useradd -r -g researcher -m -s /bin/bash researcher
WORKDIR /workspace
RUN chown researcher:researcher /workspace

# Entrypoint script — three-phase execution (pip install → setup → experiment)
COPY entrypoint.sh /usr/local/bin/rc-entrypoint.sh
RUN chmod +x /usr/local/bin/rc-entrypoint.sh

USER researcher

# Default: run entrypoint wrapper with main.py as the experiment script
ENTRYPOINT ["/usr/local/bin/rc-entrypoint.sh"]
CMD ["main.py"]
