FROM python:3.9-slim

# Устанавливаем зависимости
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Создаем рабочую директорию
WORKDIR /app

# Клонируем репозиторий (уже с измененным файлом)
RUN git clone https://github.com/Den2909/multi_class . && \
    rm -rf .git

# Устанавливаем зависимости Python
RUN pip install --no-cache-dir -r requirements_app.txt

# Открываем порт, который использует FastAPI
EXPOSE 8000

# Запускаем приложение
CMD ["uvicorn", "app_api_web:app", "--host", "0.0.0.0", "--port", "8000"]
