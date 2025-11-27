FROM python:3.12-slim

# Install curl and build dependencies for psutil
RUN apt-get update && apt-get install -y curl gcc python3-dev

# Install poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Add poetry to PATH
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Copy poetry config files
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root

# Copy project files
COPY . ./

WORKDIR /app

# Run with poetry
WORKDIR /app
CMD ["poetry", "run", "streamlit", "run", "1_general_dashboard.py", "--server.port=8502", "--server.address=0.0.0.0"]