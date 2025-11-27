import os
from pathlib import Path

MODEL_TEMPLATE = [
    "api/__init__.py",
    "api/routes.py",
    "api/models.py",
    "api/dependencies.py",
    "api/utils.py",
    "archive/spaceholder",
    "data/spaceholder",
    "service/src/dashboard.py",
    "service/src/model.py",
    "service/src/preprocess.py",
    "service/src/train.py",
    "service/main.py",
    "Dockerfile",
    "pyproject.toml",
    "poetry.lock",
]

README_TEMPLATE = """# New Model

This is a placeholder README for your new model implementation.
You can customize this once your model logic is ready.
"""

DOCKERFILE_TEMPLATE = """FROM python:3.10-slim

RUN apt-get update && apt-get install -y curl
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root
COPY service/ ./service/
COPY data/ ./data/
WORKDIR /app/service
CMD ["poetry", "run", "streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
"""

PYPROJECT_TEMPLATE = """[tool.poetry]
name = "new_model"
version = "0.1.0"
description = ""
authors = ["Your Name <you@example.com>"]

[tool.poetry.dependencies]
python = "^3.10"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
"""

def create_model_structure(model_name):
    model_path = Path(model_name)
    if model_path.exists():
        print(f"❌ Model '{model_name}' already exists.")
        return

    for file in MODEL_TEMPLATE:
        file_path = model_path / file
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if not file_path.name.endswith('.keep'):
            file_path.touch()

    # Optional template files
    (model_path / "README.md").write_text(README_TEMPLATE)
    (model_path / "Dockerfile").write_text(DOCKERFILE_TEMPLATE)
    (model_path / "pyproject.toml").write_text(PYPROJECT_TEMPLATE)
    (model_path / "poetry.lock").touch()

    print(f"Model '{model_name}' structure created.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python setup_newModel.py <model_name>")
    else:
        create_model_structure(sys.argv[1])