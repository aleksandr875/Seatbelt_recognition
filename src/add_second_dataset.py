from pathlib import Path
import random
import shutil


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Новый скачанный датасет safedriving-seatbelt
SOURCE_DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "safedriving_seatbelt"
SOURCE_IMAGES_DIR = SOURCE_DATASET_DIR / "train" / "images"
SOURCE_LABELS_DIR = SOURCE_DATASET_DIR / "train" / "labels"

# Старый классификационный датасет
OLD_DATASET_DIR = PROJECT_ROOT / "data" / "processed"

# Класс из safedriving:
# 0 — person-noseatbelt
TARGET_CLASS_ID = 0

# В старом датасете этот класс соответствует belt_off
TARGET_CLASS_NAME = "belt_off"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

FILENAME_PREFIX = "safedriving_"


def label_has_target_class(label_path: Path) -> bool:
    """
    Проверяет, есть ли в YOLO label-файле нужный класс.
    В нашем случае нужен class_id = 0, то есть person-noseatbelt.
    """

    if not label_path.exists():
        return False

    lines = label_path.read_text(encoding="utf-8").strip().splitlines()

    for line in lines:
        parts = line.strip().split()

        if not parts:
            continue

        try:
            class_id = int(parts[0])
        except ValueError:
            continue

        if class_id == TARGET_CLASS_ID:
            return True

    return False


def collect_no_belt_images():
    """
    Собирает изображения, в которых есть класс person-noseatbelt.
    """

    images = [
        path for path in SOURCE_IMAGES_DIR.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    selected_images = []

    for image_path in images:
        label_path = SOURCE_LABELS_DIR / f"{image_path.stem}.txt"

        if label_has_target_class(label_path):
            selected_images.append(image_path)

    return selected_images


def split_images(images):
    """
    Делит изображения на train / val / test.
    """

    random.seed(42)
    random.shuffle(images)

    total = len(images)
    train_count = int(total * TRAIN_RATIO)
    val_count = int(total * VAL_RATIO)

    train_images = images[:train_count]
    val_images = images[train_count:train_count + val_count]
    test_images = images[train_count + val_count:]

    return {
        "train": train_images,
        "val": val_images,
        "test": test_images,
    }


def get_unique_target_path(target_dir: Path, original_name: str, index: int) -> Path:
    """
    Создает уникальное имя файла с префиксом safedriving_,
    чтобы не перезаписать старые изображения.
    """

    suffix = Path(original_name).suffix.lower()
    new_name = f"{FILENAME_PREFIX}{index:06d}{suffix}"
    target_path = target_dir / new_name

    counter = 1

    while target_path.exists():
        new_name = f"{FILENAME_PREFIX}{index:06d}_{counter}{suffix}"
        target_path = target_dir / new_name
        counter += 1

    return target_path


def copy_images_to_old_dataset(split_images_dict):
    """
    Копирует выбранные изображения в старый датасет:
    data/processed/train/belt_off
    data/processed/val/belt_off
    data/processed/test/belt_off
    """

    global_index = 1

    for split, images in split_images_dict.items():
        target_dir = OLD_DATASET_DIR / split / TARGET_CLASS_NAME
        target_dir.mkdir(parents=True, exist_ok=True)

        copied = 0

        for image_path in images:
            target_path = get_unique_target_path(
                target_dir=target_dir,
                original_name=image_path.name,
                index=global_index
            )

            shutil.copy2(image_path, target_path)

            copied += 1
            global_index += 1

        print(f"{split}: добавлено {copied} изображений в {target_dir}")


def main():
    if not SOURCE_IMAGES_DIR.exists():
        raise FileNotFoundError(f"Не найдена папка с изображениями: {SOURCE_IMAGES_DIR}")

    if not SOURCE_LABELS_DIR.exists():
        raise FileNotFoundError(f"Не найдена папка с label-файлами: {SOURCE_LABELS_DIR}")

    if not OLD_DATASET_DIR.exists():
        raise FileNotFoundError(f"Не найден старый датасет: {OLD_DATASET_DIR}")

    selected_images = collect_no_belt_images()

    print(f"Найдено изображений person-noseatbelt: {len(selected_images)}")

    if not selected_images:
        print("Ничего не добавлено.")
        return

    split_images_dict = split_images(selected_images)

    print("Распределение добавляемых изображений:")
    print(f"Train: {len(split_images_dict['train'])}")
    print(f"Val:   {len(split_images_dict['val'])}")
    print(f"Test:  {len(split_images_dict['test'])}")

    copy_images_to_old_dataset(split_images_dict)

    print()
    print("Добавление изображений в старый датасет завершено.")
    print("Теперь можно вручную чистить папки belt_off в train / val / test.")


if __name__ == "__main__":
    main()