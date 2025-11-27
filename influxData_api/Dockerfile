FROM python:3.12-slim

# Install curl (needed for poetry installation)
RUN apt-get update && apt-get install -y curl

# Install poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Add poetry to PATH
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app
ENV PYTHONPATH=/app

# Copy poetry config files
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root

# Copy project files
COPY . ./

# Expose port
EXPOSE 8009

# Start FastAPI server
CMD ["poetry", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8009"]
