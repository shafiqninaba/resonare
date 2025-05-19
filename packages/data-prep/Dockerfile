FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV PATH="/app/.venv/bin:$PATH"

# For now, no cache mounts as we run out of disk space in uv/.cache
# RUN --mount=type=cache,target=/root/.cache/uv \
#     --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
#     uv sync --no-install-project

RUN --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --no-install-project --no-cache

ADD . /app

# RUN --mount=type=cache,target=/root/.cache/uv \
#     uv sync --frozen --no-dev

RUN uv sync --frozen --no-dev --no-cache

CMD ["fastapi", "run", "src/app/main.py"]