# Default is a clean pinned base. An existing project image can be reused for
# dependency-only updates when its Python and OS versions remain unchanged.
ARG BASE_IMAGE=python:3.11.14-slim-bookworm
FROM ${BASE_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYGAME_HIDE_SUPPORT_PROMPT=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates git ffmpeg libgl1 libegl1 libglib2.0-0 tk8.6 \
    libx11-6 libxext6 libxrender1 libxkbcommon0 libwayland-client0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements.lock /opt/deeprl/requirements.lock
ARG PIP_INDEX_URL=https://pypi.org/simple
RUN --mount=type=cache,target=/root/.cache/uv \
    python -m pip install --no-cache-dir --index-url "$PIP_INDEX_URL" uv==0.9.5 \
    && UV_CONCURRENT_DOWNLOADS=4 UV_LINK_MODE=copy uv pip install --system --no-deps \
        --index-url "$PIP_INDEX_URL" -r /opt/deeprl/requirements.lock \
    && python -m pip check \
    && tensorboard --version

# Source and outputs are bind-mounted at runtime; code edits need no rebuild.
CMD ["bash"]
