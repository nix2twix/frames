# facenet_arch.py
# Файл архитектуры для модели FaceNet (InceptionResnetV1) из библиотеки facenet-pytorch.

import torch.nn as nn
# Импортируем оригинальный класс из библиотеки.
# Важно: пользователь, загружающий этот файл, тоже должен иметь установленный facenet-pytorch.
from facenet_pytorch import InceptionResnetV1

class Embedder(InceptionResnetV1):
    """
    Наследуемся от InceptionResnetV1, чтобы ключи state_dict 
    совпадали (без префикса model.).
    """
    def __init__(self):
        super().__init__(pretrained=None, classify=False)