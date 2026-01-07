FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS base
ENV PATH="/app/.venv/bin:$PATH"
WORKDIR /app

FROM base AS build_stage
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

FROM base AS runtime
COPY src/ src/
COPY configs/ configs
COPY components/ components/
COPY utils/ utils/
COPY models/ models/

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

ENV PYTHONPATH=/app

CMD ["bash"]