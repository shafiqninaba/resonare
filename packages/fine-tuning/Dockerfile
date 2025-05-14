FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    pkg-config \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install cusparselt
RUN wget https://developer.download.nvidia.com/compute/cusparselt/0.7.1/local_installers/cusparselt-local-repo-ubuntu2204-0.7.1_1.0-1_amd64.deb && \
    dpkg -i cusparselt-local-repo-ubuntu2204-0.7.1_1.0-1_amd64.deb && \
    cp /var/cusparselt-local-repo-ubuntu2204-0.7.1/cusparselt-*-keyring.gpg /usr/share/keyrings/ && \
    apt-get update && apt-get -y install libcusparselt0 libcusparselt-dev && \
    rm cusparselt-local-repo-ubuntu2204-0.7.1_1.0-1_amd64.deb

# Install UV
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the PATH
ENV PATH="/root/.local/bin/:$PATH"

# Set working directory
WORKDIR /app

# Set environment variables
ENV UV_COMPILE_BYTECODE=1
ENV PYTHONIOENCODING=utf8
ENV LANG="C.UTF-8"
ENV LC_ALL="C.UTF-8"
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV UV_LINK_MODE=copy

# Create virtual environment
RUN uv venv --python 3.12

# Install dependencies using cache mounts
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --no-install-project

# Add project source code and install it
ADD . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

CMD ["fastapi", "run", "main.py"]
