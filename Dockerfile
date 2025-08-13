# Используем официальный образ Python
FROM python:3.9-slim

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Создаем не-root пользователя и директорию /app
RUN useradd -m appuser && \
    mkdir -p /app && \
    chown appuser:appuser /app

# Переключаемся на созданного пользователя
USER appuser
WORKDIR /app

# Клонируем репозиторий
RUN git clone https://github.com/Den2909/multi_class . && \
    rm -rf .git

# Создаем и активируем виртуальное окружение
RUN python -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Обновляем pip в виртуальном окружении
RUN pip install --upgrade pip

# Устанавливаем зависимости Python
RUN if [ -f requirements_app.txt ]; then pip install --no-cache-dir -r requirements_app.txt; else echo "No requirements file found"; fi

# Открываем порт для FastAPI
EXPOSE 8000

# Проверяем здоровье приложения
HEALTHCHECK --interval=30s --timeout=3s CMD curl -f http://localhost:8000/ || exit 1

# Запускаем приложение
CMD ["uvicorn", "app_api_web:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]