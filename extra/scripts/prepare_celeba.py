import os
import zipfile
from collections import defaultdict
from io import BytesIO
from PIL import Image

# --- НАСТРОЙКИ ---
DATA_ROOT = r"D:\Yandex.Disk\PROJECTS\FRAMES\data"
ZIP_PATH = os.path.join(DATA_ROOT, "img_align_celeba.zip")
INFO_FILE = os.path.join(DATA_ROOT, "identity_CelebA.txt")
OUTPUT_ROOT = os.path.join(DATA_ROOT, "celeba_subset")

N_IDENTITIES = 30
MIN_IMAGES_PER_ID = 5
MAX_IMAGES_PER_ID = 20

# --- Проверки ---
assert os.path.isfile(ZIP_PATH), f"Нет архива: {ZIP_PATH}"
assert os.path.isfile(INFO_FILE), f"Нет файла identity: {INFO_FILE}"

# --- Читаем identity ---
identity_map = defaultdict(list)
with open(INFO_FILE, encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) != 2:
            continue
        fname, ident = parts
        identity_map[int(ident)].append(fname)

print(f"Всего личностей в файле: {len(identity_map)}")

# --- Отбираем личности с достаточным числом фото ---
candidates = [
    (ident, files)
    for ident, files in identity_map.items()
    if MIN_IMAGES_PER_ID <= len(files) <= MAX_IMAGES_PER_ID
]
candidates.sort(key=lambda x: x[0])
selected = candidates[:N_IDENTITIES]

print(f"Отобрано личностей: {len(selected)}")

if not selected:
    print("Ничего не найдено. Уменьшите MIN_IMAGES_PER_ID.")
    raise SystemExit(1)

# --- Готовим выходную папку ---
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# --- Открываем ZIP и извлекаем только нужное ---
# Внутри архива структура: img_align_celeba/000001.jpg
PREFIX = "img_align_celeba/"

copied_total = 0
missing_total = 0

with zipfile.ZipFile(ZIP_PATH, "r") as zf:
    # Соберём множество всех файлов в архиве (для быстрой проверки)
    # Это O(N) по числу файлов, но не требует распаковки
    print("Индексирую содержимое архива...")
    archive_names = set(zf.namelist())
    print(f"В архиве файлов: {len(archive_names)}")

    for idx, (ident, files) in enumerate(selected):
        class_dir = os.path.join(OUTPUT_ROOT, f"person_{idx:03d}")
        os.makedirs(class_dir, exist_ok=True)

        copied = 0
        for fname in files[:MAX_IMAGES_PER_ID]:
            archive_path = PREFIX + fname
            if archive_path not in archive_names:
                missing_total += 1
                print(f"  ⚠ Нет в архиве: {archive_path}")
                continue

            # Извлекаем один файл
            data = zf.read(archive_path)
            dst = os.path.join(class_dir, fname)
            with open(dst, "wb") as fout:
                fout.write(data)
            copied += 1

        copied_total += copied
        print(f"  person_{idx:03d}: {copied} фото (identity={ident})")

print()
print(f"Готово: {OUTPUT_ROOT}")
print(f"Извлечено файлов: {copied_total}")
print(f"Пропущено: {missing_total}")