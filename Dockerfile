FROM python:3.11-slim

# Pin the uv version (use the one from `uv --version`) rather than :latest
COPY --from=ghcr.io/astral-sh/uv:0.12.19 /uv /bin/uv

WORKDIR /app

# Layer 1: dependencies only (cached until the lockfile changes)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project --no-editable

# Layer 2: the package itself (rebuilt on every code change, but fast)
COPY src/ src/
RUN uv sync --frozen --no-dev --no-editable

COPY models/model.joblib models/model.joblib
ENV MODEL_PATH=/app/models/model.joblib
ENV PATH="/app/.venv/bin:$PATH"

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "titanic.api:app", "--host", "0.0.0.0", "--port", "8000"]