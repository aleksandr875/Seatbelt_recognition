from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "processed"

SPLITS = ["train", "val", "test"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def count_images_in_dir(directory: Path):
    if not directory.exists():
        return 0

    return len([
        file for file in directory.iterdir()
        if file.suffix.lower() in IMAGE_EXTENSIONS
    ])


def get_classes():
    train_dir = DATASET_DIR / "train"

    if not train_dir.exists():
        raise FileNotFoundError(f"Папка train не найдена: {train_dir}")

    classes = [
        path.name for path in train_dir.iterdir()
        if path.is_dir()
    ]

    classes.sort()

    return classes


def main():
    classes = get_classes()

    print(f"Классы: {classes}")

    total_all = 0
    class_totals = {class_name: 0 for class_name in classes}

    for split in SPLITS:
        print()
        print(split.upper())

        split_total = 0

        for class_name in classes:
            class_dir = DATASET_DIR / split / class_name
            count = count_images_in_dir(class_dir)

            split_total += count
            class_totals[class_name] += count

            print(f"{class_name}: {count}")

        total_all += split_total
        print(f"Итого {split}: {split_total}")

    print()
    print("Итого по классам:")

    for class_name, count in class_totals.items():
        print(f"{class_name}: {count}")

    print()
    print(f"Всего изображений: {total_all}")


if __name__ == "__main__":
    main()