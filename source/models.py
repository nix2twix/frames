# source/models.py
import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1
import os
import sys
import importlib.util

# ---------------------------
# FaceNet (InceptionResnetV1)
# ---------------------------
class FaceNetEmbedder(nn.Module):
    def __init__(self, pretrained='vggface2', device=None):
        super().__init__()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = InceptionResnetV1(pretrained=pretrained).eval().to(self.device)

    def forward(self, x):
        return self.model(x)

    def get_embeddings(self, x):
        with torch.no_grad():
            return self.model(x)


# ---------------------------
# Универсальный выбор модели
# ---------------------------
def get_embedder(model_name: str, device=None):
    name = model_name.lower()
    if name == "facenet":
        return FaceNetEmbedder(device=device)
    else:
        raise ValueError(f"Unknown model {model_name}")


class CustomEmbedder(nn.Module):
    """
    PyTorch: state_dict (.pt / .pth) + файл архитектуры (.py).
    Архитектура импортируется из файла, создается экземпляр,
    в него загружается state_dict через torch.load(..., weights_only=True).
    """

    def __init__(
        self,
        model_path: str,
        framework: str = "pytorch",
        arch_path: str = None,
        class_name: str = None,
        device=None,
    ):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.framework = framework.lower()
        self.model = None

        if self.framework == "pytorch":
            self._load_pytorch(model_path, arch_path, class_name)
        else:
            raise ValueError(f"Неподдерживаемый фреймворк: {framework}")
    
    # ---------- PyTorch ----------
    def _load_pytorch(self, model_path: str, arch_path: str, class_name: str):
        if arch_path is None or class_name is None:
            raise ValueError(
                "Для загрузки PyTorch-модели необходимо указать файл архитектуры "
                "(.py) и имя класса модели."
            )

        try:
            state_dict = torch.load(
                model_path,
                map_location=self.device,
                weights_only=True,
            )
        except Exception as e:
            raise RuntimeError(
                f"Не удалось безопасно загрузить state_dict: {e}\n"
                "Убедитесь, что файл содержит только веса (state_dict), "
                "а не полную модель."
            ) from e

        if not isinstance(state_dict, dict):
            raise ValueError(
                "Файл не является state_dict. Загрузка полных моделей "
                "запрещена по соображениям безопасности."
            )

        # Импорт архитектуры из .py-файла
        ModelClass = self._import_class_from_file(arch_path, class_name)

        # Создание модели и загрузка весов
        model = ModelClass()
        try:
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"[CustomEmbedder] Отсутствуют ключи: {missing}")
            if unexpected:
                print(f"[CustomEmbedder] Лишние ключи проигнорированы: {unexpected}")
        except Exception as e:
            raise RuntimeError(
                f"Не удалось загрузить веса в модель '{class_name}': {e}"
            ) from e

        self.model = model.eval().to(self.device)

    @staticmethod
    def _import_class_from_file(py_path: str, class_name: str):
        """Импортирует класс из указанного .py-файла."""
        if not os.path.isfile(py_path):
            raise FileNotFoundError(f"Файл архитектуры не найден: {py_path}")

        spec = importlib.util.spec_from_file_location("user_model", py_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["user_model"] = module
        spec.loader.exec_module(module)

        if not hasattr(module, class_name):
            raise AttributeError(
                f"В файле {py_path} не найден класс '{class_name}'"
            )
        return getattr(module, class_name)

    # ---------- Inference ----------
    def forward(self, x):
        return self.model(x)

    def get_embeddings(self, x):
        with torch.no_grad():
            return self.forward(x)
