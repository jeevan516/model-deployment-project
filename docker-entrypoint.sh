#!/bin/sh
# Train the model on first start if none is mounted, then serve it.
set -e
uv run --no-sync python -m src.prepare_model
exec uv run --no-sync uvicorn app.main:app --host 0.0.0.0 --port 8000
