FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

# Change the working directory to the `app` directory
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Changing the default UV_LINK_MODE silences warnings about not being able to use hard links since the cache and sync target are on separate file systems.
ENV UV_LINK_MODE=copy

# Install dependencies
# Ren Hwa: by right, uv lockfile should be included in the repo. For nowm, we just use the toml file to install the dependencies.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --no-install-project

# Copy the project into the image
ADD . /app

# Sync the project
# Locked -> To disable automatic locking, use the --locked option: If the lockfile is not up-to-date, uv will raise an error instead of updating the lockfile.
# Frozen -> To use the lockfile without checking if it is up-to-date, use the --frozen option
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

CMD ["python", "app.py"]
