FROM python:3.12-slim

# Install curl (needed for poetry installation)
RUN apt-get update && apt-get install -y curl

# Install poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Add poetry to PATH
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Copy poetry config files
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root

# Copy project files
COPY service/ ./service/

WORKDIR /app/service

# Run with poetry
WORKDIR /app/service
CMD ["poetry", "run", "streamlit", "run", "src/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]