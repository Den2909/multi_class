## Инструкция по запуску Docker-контейнера самостоятельно (необходим токен для Telegram)

1. **Клонируйте репозиторий**:
   ```bash
   sudo apt update
   sudo apt install git -y
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
Дополнительный команды:
Установка git
   ```bash
   sudo apt update
   sudo apt install git -y
   ```
Установка docker
   ```bash
   # Установим зависимости
sudo apt update
sudo apt install ca-certificates curl -y

# Добавим официальный GPG ключ Docker
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Добавим репозиторий
echo \
  "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Обновим пакеты
sudo apt update
   ```

 ```bash
sudo apt install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin -y
   ```
Добавим пользователя в группу docker:
 ```bash
sudo usermod -aG docker multi_class
   ```

Проверим установку (необязательно):
 ```bash
sudo docker --version
sudo docker run hello-world
   ```
После установки docker лучше перезапустить сессию: exit

Удаление неиспользуемых контейнеров, образы и тома:
 ```bash
docker rm -f multi_class_container
docker image prune -a -f
docker volume prune -f
docker system prune -a --volumes -f
   ```

