## Инструкция по запуску Docker-контейнера самостоятельно (необходим токен для Telegram)

1. **Клонируйте репозиторий**:
   ```bash
   git clone https://github.com/Den2909/multi_class
   cd multi_class
   ```

2. **Соберите Docker-образ**:
   ```bash
   docker build --no-cache -t multi_class_app .
   ```

3. **Запустите контейнер**:

  
     ```bash
     docker run -d -p 8000:8000 --name multi_class_container --restart=always multi_class_app
     ```

   остановка контейнера
     ```bash
     docker rm -f multi_class_container
     ```
