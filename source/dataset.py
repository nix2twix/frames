import zipfile
import tempfile
import torch
import numpy as np
from pathlib import Path
from facenet_pytorch import MTCNN
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class subsetDataset(Dataset):
    def __init__(self, root, transform=None, aligned=False):
        self.root = root
        self.transform = transform or transforms.ToTensor()
        self.aligned = aligned
        self.dataset = datasets.ImageFolder(root=root)
        self.samples = self.dataset.samples
        self.classes = self.dataset.classes
        if not aligned:
            self.mtcnn = MTCNN(image_size=112)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.aligned:
            img = self.transform(img)
        else:
            img = self.mtcnn(img)  # MTCNN вернёт тензор C×H×W или None
            if img is None:
                # Если лицо не найдено, возвращаем чёрный тензор
                img = torch.zeros(3, 112, 112)
            elif isinstance(img, Image.Image) or isinstance(img, np.ndarray):
                # На случай, если MTCNN возвращает PIL Image
                img = transforms.ToTensor()(img)
            # иначе img уже тензор — оставляем как есть
        return img, label


    def __len__(self):
        return len(self.samples)

def get_dataloader(root, batch_size=8, shuffle=True, aligned=False):
    ds = subsetDataset(root=root, aligned=aligned)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)

# --- ZIP загрузка и распаковка ---
def load_dataset_from_zip(zip_file) -> str:
    """
    Принимает загруженный через Streamlit zip_file,
    распаковывает его во временную папку и возвращает путь к ней.
    """
    temp_dir = tempfile.mkdtemp(prefix="adverslab_zip_")
    with zipfile.ZipFile(zip_file) as zf:
        zf.extractall(temp_dir)
    return temp_dir

def validate_data_root(root_path):
    """
    Проверяет, есть ли в папке подкаталоги с изображениями и что они не пустые.
    Возвращает dict:
    {
        "ok": bool,
        "summary": str,
        "classes": list of class names,
        "empty_folders": list of folders,
        "bad_files": list of file paths
    }
    """
    path = Path(root_path)
    result = {"ok": False, "summary": "", "classes": [], "empty_folders": [], "bad_files": []}
    if not path.exists():
        result["summary"] = "Путь не существует!"
        return result
    if not path.is_dir():
        result["summary"] = "Путь не является папкой!"
        return result

    class_dirs = [p for p in path.iterdir() if p.is_dir()]
    empty = [str(p) for p in class_dirs if not any(p.iterdir())]
    bad_files = []
    for d in class_dirs:
        for f in d.iterdir():
            if f.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                bad_files.append(str(f))
    result["classes"] = [d.name for d in class_dirs]
    result["empty_folders"] = empty
    result["bad_files"] = bad_files

    if not class_dirs:
        result["summary"] = "В папке нет подкаталогов с изображениями!"
    elif empty or bad_files:
        result["summary"] = f"Найдены проблемы: {len(empty)} пустых папок, {len(bad_files)} неподходящих файлов."
    else:
        result["summary"] = f"Все в порядке. Найдено {len(class_dirs)} классов."
        result["ok"] = True

    return result
