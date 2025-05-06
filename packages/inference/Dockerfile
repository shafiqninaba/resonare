FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

## INSTALL SYSTEM DEPENDENCIES ##
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    pkg-config \
    cmake \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://developer.download.nvidia.com/compute/cusparselt/0.7.1/local_installers/cusparselt-local-repo-ubuntu2204-0.7.1_1.0-1_amd64.deb
RUN dpkg -i cusparselt-local-repo-ubuntu2204-0.7.1_1.0-1_amd64.deb
RUN cp /var/cusparselt-local-repo-ubuntu2204-0.7.1/cusparselt-*-keyring.gpg /usr/share/keyrings/
RUN apt-get update && apt-get -y install libcusparselt0 libcusparselt-dev


## INSTALL UV ##
# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"
#################

# Install the project into `/app`
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1
ENV PYTHONIOENCODING=utf8
ENV LANG="C.UTF-8"
ENV LC_ALL="C.UTF-8"
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

RUN uv venv --python 3.12

# Install the project's dependencies using the lockfile and settings
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --no-install-project

# Then, add the rest of the project source code and install it
# Installing separately from its dependencies allows optimal layer caching
ADD . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

CMD ["fastapi", "run", "--host", "0.0.0.0", "main.py"]
