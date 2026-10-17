FROM python:3.11-slim

WORKDIR /app

# Install the package and its main dependencies (no train/dev extras)
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir .

# The trained model, and where the code should look for it
COPY models/model.joblib models/model.joblib
ENV MODEL_PATH=/app/models/model.joblib

# Don't run as root
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "titanic.api:app", "--host", "0.0.0.0", "--port", "8000"]