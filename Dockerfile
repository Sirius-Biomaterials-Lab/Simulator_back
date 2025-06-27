FROM python:3.12.1-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml poetry.lock* ./

# Установка Poetry
RUN pip install --no-cache-dir poetry==1.8.3\
    && python -m poetry config virtualenvs.create false \
    && python -m poetry install --no-interaction --no-ansi --without dev --no-root

#COPY . /app

#CMD ["poetry", "run", "uvicorn", "--factory", "app.main:create_app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
CMD ["poetry", "run", "uvicorn", "--factory", "app.main:create_app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
