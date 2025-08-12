# Базовый образ Python
FROM python:3.9-slim
FROM python:3.9-slim

# Установка Git для клонирования репозитория
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Клонирование репозитория
RUN git clone https://github.com/Den2909/multi_class /app
# Рабочая директория
WORKDIR /app

# Копируем все файлы  в контейнер
COPY . /app

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Определяем команду запуска: uvicorn с хостом 0.0.0.0 для доступа снаружи, порт 8000
CMD ["uvicorn", "app_api_web:app", "--host", "0.0.0.0", "--port", "8000"]
