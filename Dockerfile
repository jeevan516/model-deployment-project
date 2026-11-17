FROM ghcr.io/astral-sh/uv:0.13.5 AS uv
FROM python:3.13.17-slim-bookworm

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY --from=uv /uv /uvx /bin/
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project --no-cache

# Code only. The model comes from ./artifacts (mounted) or a volume; if it is missing,
# the entrypoint downloads January 2025 and trains it once (about a minute).
COPY src ./src
COPY app ./app
COPY docker-entrypoint.sh ./
ENV MODEL_PATH=/app/artifacts/model.skops
EXPOSE 8000
CMD ["./docker-entrypoint.sh"]
