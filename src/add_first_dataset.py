from pathlib import Path
import random
import shutil
from collections import defaultdict


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "seatbelt_yolov8"
RAW_IMAGES_DIR = RAW_DATASET_DIR / "train" / "images"
RAW_LABELS_DIR = RAW_DATASET_DIR / "train" / "labels"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

CLASS_MAP = {
    0: "belt_off",   # no_seat_belt
    1: "belt_on",    # seatbelt
}

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def clean_output_dir():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    for split in ["train", "val", "test"]:
        for class_name in CLASS_MAP.values():
            (OUTPUT_DIR / split / class_name).mkdir(parents=True, exist_ok=True)


def read_label_class(label_path: Path):
    """
    Читает YOLO label-файл и возвращает класс изображения.

    В YOLO-разметке каждая строка обычно выглядит так:
    class_id x_center y_center width height

    Для классификации нам нужен только class_id.
    Если в файле несколько разных классов, изображение пропускается как неоднозначное.
    """
    if not label_path.exists():
        return None

    lines = label_path.read_text(encoding="utf-8").strip().splitlines()

    if not lines:
        return None

    class_ids = set()

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue

        try:
            class_id = int(parts[0])
            class_ids.add(class_id)
        except ValueError:
            continue

    if len(class_ids) != 1:
        return None

    class_id = class_ids.pop()

    return CLASS_MAP.get(class_id)


def collect_images_by_class():
    images_by_class = defaultdict(list)

    all_images = [
        path for path in RAW_IMAGES_DIR.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    skipped = 0

    for image_path in all_images:
        label_path = RAW_LABELS_DIR / f"{image_path.stem}.txt"
        class_name = read_label_class(label_path)

        if class_name is None:
            skipped += 1
            continue

        images_by_class[class_name].append(image_path)

    return images_by_class, skipped


def split_list(items):
    random.shuffle(items)

    total = len(items)
    train_count = int(total * TRAIN_RATIO)
    val_count = int(total * VAL_RATIO)

    train_items = items[:train_count]
    val_items = items[train_count:train_count + val_count]
    test_items = items[train_count + val_count:]

    return train_items, val_items, test_items


def copy_images(images, split, class_name):
    target_dir = OUTPUT_DIR / split / class_name

    for image_path in images:
        target_path = target_dir / image_path.name
        shutil.copy2(image_path, target_path)


def main():
    if not RAW_IMAGES_DIR.exists():
        raise FileNotFoundError(f"Не найдена папка с изображениями: {RAW_IMAGES_DIR}")

    if not RAW_LABELS_DIR.exists():
        raise FileNotFoundError(f"Не найдена папка с label-файлами: {RAW_LABELS_DIR}")

    random.seed(42)

    clean_output_dir()

    images_by_class, skipped = collect_images_by_class()

    print("Найдено изображений по классам:")

    total_used = 0

    for class_name, images in images_by_class.items():
        print(f"{class_name}: {len(images)}")
        total_used += len(images)

    print(f"Пропущено неоднозначных или некорректных изображений: {skipped}")
    print(f"Всего используется изображений: {total_used}")

    for class_name, images in images_by_class.items():
        train_images, val_images, test_images = split_list(images)

        copy_images(train_images, "train", class_name)
        copy_images(val_images, "val", class_name)
        copy_images(test_images, "test", class_name)

        print()
        print(f"Класс: {class_name}")
        print(f"Train: {len(train_images)}")
        print(f"Val:   {len(val_images)}")
        print(f"Test:  {len(test_images)}")

    print()
    print("Классификационный датасет подготовлен.")
    print(f"Папка результата: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()