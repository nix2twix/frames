import torch
from facenet_pytorch import InceptionResnetV1
import os

# 1. Создаём модель. Это автоматически скачает веса в кеш torch.
print("Загрузка модели и весов...")
model = InceptionResnetV1(pretrained='vggface2')

# 2. Определяем путь для сохранения в вашем проекте
output_dir = "downloaded_models"
os.makedirs(output_dir, exist_ok=True)
weights_path = os.path.join(output_dir, "facenet_vggface2_state_dict.pth")

# 3. Сохраняем state_dict (только веса, без архитектуры)
torch.save(model.state_dict(), weights_path)
print(f"Веса сохранены в: {weights_path}")

# Проверка: посмотрим на структуру state_dict
print(f"Количество параметров: {len(model.state_dict())}")