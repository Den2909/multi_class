import os
import torch
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from torch import nn
import torch.nn.functional as F
from io import BytesIO
import base64

# === Конфигурация ===
root_data_dir = '/mnt/d/Python/Project/Nikitenko_multi_class/DataSet_V3/'
class_names = sorted(os.listdir(root_data_dir))
num_of_classes = len(class_names)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Модуль канального внимания ===
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, in_planes // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_planes // ratio, in_planes, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).squeeze(-1).squeeze(-1))
        max_out = self.fc(self.max_pool(x).squeeze(-1).squeeze(-1))
        out = avg_out + max_out
        return self.sigmoid(out).unsqueeze(-1).unsqueeze(-1)

# === Модуль пространственного внимания ===
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.conv2 = nn.Conv2d(1, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.norm = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        x = self.norm(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        return self.sigmoid(x)

# === CBAM (Convolutional Block Attention Module) ===
class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

# === Модель ===
class MyNetWithCBAM(nn.Module):
    def __init__(self, in_channels: int = 3, num_of_classes: int = len(class_names)):
        super(MyNetWithCBAM, self).__init__()
        self.efficient_net = EfficientNet.from_pretrained('efficientnet-b1')
        in_features_efficient_net = self.efficient_net._fc.in_features
        self.efficient_net._fc = nn.Identity()
        self.cbam = CBAM(in_planes=self.efficient_net._conv_head.out_channels)
        self.base_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features_efficient_net, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_of_classes)
        )

    def forward(self, x):
        x_efficient_net = self.efficient_net.extract_features(x)
        x_efficient_net = self.cbam(x_efficient_net)
        x_efficient_net = x_efficient_net.mean([2, 3])
        x_base_model = self.base_classifier(x_efficient_net)
        x = self.classifier(x_base_model)
        return x

# === Инициализация модели ===
model = MyNetWithCBAM(num_of_classes=num_of_classes).to(device)
checkpoint = torch.load("/mnt/d/Python/Project/Nikitenko_multi_class/models/v4/checkpoint.pth", map_location=device, weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# === Преобразования ===
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]),
])

# === FastAPI приложение ===
app = FastAPI(title="Image Classification API and Web Interface")

# === CORS (для API) ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене замените на конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Подключение шаблонов Jinja2 ===
templates = Jinja2Templates(directory="templates")

# === Функция предсказания ===
def predict(image: Image.Image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        probs = F.softmax(output[0], dim=0)
        predicted_class = torch.argmax(probs).item()
        label = class_names[predicted_class]
        return label, probs[predicted_class].item()

# === Функция для преобразования изображения в base64 ===
def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

# === Главная страница (веб-интерфейс) ===
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# === Веб-обработка изображения (результат на странице) ===
@app.post("/predict", response_class=HTMLResponse)
async def predict_image_web(request: Request, file: UploadFile = File(...)):
    if not file.filename:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": "No file uploaded"},
            status_code=400
        )

    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        label, confidence = predict(image)
        img_base64 = image_to_base64(image)
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "predicted_class": label,
                "confidence": round(confidence, 4),
                "image": img_base64
            }
        )
    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": str(e)},
            status_code=500
        )

# === API-эндпоинт (JSON-ответ) ===
@app.post("/api/predict")
async def predict_image_api(file: UploadFile = File(...)):
    if not file.filename:
        return JSONResponse(status_code=400, content={"error": "No file uploaded"})

    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        label, confidence = predict(image)
        return {"predicted_class": label, "confidence": round(confidence, 4)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# === Запуск приложения ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)