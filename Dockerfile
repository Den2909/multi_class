# Используем официальный образ Python
FROM python:3.9-slim

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    libgl1-mesa-dri \
    libglib2.0-0 \
    curl \
    build-essential \
    python3-dev \
    libpng-dev \
    libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

# Создаем не-root пользователя и директорию /app
RUN useradd -m appuser && \
    mkdir -p /app && \
    chown appuser:appuser /app

# Переключаемся на созданного пользователя
USER appuser
WORKDIR /app

# Копируем локальные файлы
COPY . /app

# Создаем и активируем виртуальное окружение
RUN python -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Обновляем pip и устанавливаем базовые зависимости
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir uvicorn fastapi efficientnet-pytorch pillow jinja2 torchvision python-multipart
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Устанавливаем зависимости Python из requirements_app.txt (если есть)
RUN if [ -f requirements_app.txt ]; then pip install --no-cache-dir -r requirements_app.txt; else echo "No requirements file found"; fi

# Открываем порт для FastAPI
EXPOSE 8000

# Проверяем здоровье приложения
HEALTHCHECK --interval=30s --timeout=3s CMD curl -f http://localhost:8000/ || exit 1

# Запускаем приложение
CMD ["uvicorn", "app_api_web:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
