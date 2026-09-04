from pathlib import Path
import shutil
import csv

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_PATH = PROJECT_ROOT / "models" / "best_mobilenetv3.pt"

OUTPUT_DIR = PROJECT_ROOT / "results" / "prediction_examples"

BATCH_SIZE = 16
IMAGE_SIZE = 224

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ImageFolderWithPaths(datasets.ImageFolder):
    """
    Обычный ImageFolder возвращает только изображение и метку класса.
    Этот класс дополнительно возвращает путь к исходному файлу,
    чтобы потом можно было скопировать правильно и ошибочно
    распознанные изображения в отдельные папки.
    """

    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        path = self.samples[index][0]
        return image, label, path


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_eval_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def create_model(num_classes, device):
    """
    Создаем такую же MobileNetV3, как при обучении.
    Затем заменяем последний слой под нужное количество классов
    и загружаем сохраненные веса модели.
    """

    model = models.mobilenet_v3_small(weights=None)

    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)

    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    return model


def prepare_output_dir():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    (OUTPUT_DIR / "correct").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "wrong").mkdir(parents=True, exist_ok=True)


def make_safe_filename(text):
    """
    Убираем символы, которые могут мешать в названии файла.
    """

    return (
        text.replace("\\", "_")
        .replace("/", "_")
        .replace(":", "_")
        .replace("*", "_")
        .replace("?", "_")
        .replace('"', "_")
        .replace("<", "_")
        .replace(">", "_")
        .replace("|", "_")
    )


def copy_prediction_image(source_path, true_class, pred_class, confidence, is_correct):
    """
    Копирует исходное изображение в папку correct или wrong.

    Для правильных предсказаний:
    results/prediction_examples/correct/true_class/

    Для ошибочных предсказаний:
    results/prediction_examples/wrong/true_TRUE__pred_PRED/
    """

    source_path = Path(source_path)

    if is_correct:
        target_dir = OUTPUT_DIR / "correct" / true_class
    else:
        target_dir = OUTPUT_DIR / "wrong" / f"true_{true_class}__pred_{pred_class}"

    target_dir.mkdir(parents=True, exist_ok=True)

    confidence_text = f"{confidence:.4f}"
    new_name = f"conf_{confidence_text}__{source_path.name}"
    new_name = make_safe_filename(new_name)

    target_path = target_dir / new_name

    shutil.copy2(source_path, target_path)

    return target_path


def save_predictions_csv(rows):
    csv_path = OUTPUT_DIR / "predictions.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "source_path",
                "saved_path",
                "true_class",
                "predicted_class",
                "confidence",
                "is_correct"
            ]
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV с результатами сохранен: {csv_path}")


def main():
    device = get_device()
    print(f"Используемое устройство: {device}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Файл модели не найден: {MODEL_PATH}")

    test_dataset = ImageFolderWithPaths(
        DATA_DIR / "test",
        transform=get_eval_transform()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    class_names = test_dataset.classes

    print("Классы модели:")
    print(test_dataset.class_to_idx)

    model = create_model(
        num_classes=len(class_names),
        device=device
    )

    prepare_output_dir()

    rows = []

    correct_count = 0
    wrong_count = 0

    with torch.no_grad():
        for images, labels, paths in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)

            confidences, preds = torch.max(probabilities, dim=1)

            labels_cpu = labels.cpu().numpy()
            preds_cpu = preds.cpu().numpy()
            confidences_cpu = confidences.cpu().numpy()

            for true_label, pred_label, confidence, source_path in zip(
                labels_cpu,
                preds_cpu,
                confidences_cpu,
                paths
            ):
                true_class = class_names[true_label]
                pred_class = class_names[pred_label]

                is_correct = true_label == pred_label

                saved_path = copy_prediction_image(
                    source_path=source_path,
                    true_class=true_class,
                    pred_class=pred_class,
                    confidence=float(confidence),
                    is_correct=is_correct
                )

                rows.append({
                    "source_path": source_path,
                    "saved_path": str(saved_path),
                    "true_class": true_class,
                    "predicted_class": pred_class,
                    "confidence": float(confidence),
                    "is_correct": is_correct
                })

                if is_correct:
                    correct_count += 1
                else:
                    wrong_count += 1

    save_predictions_csv(rows)

    print()
    print("Готово.")
    print(f"Правильных предсказаний: {correct_count}")
    print(f"Ошибочных предсказаний: {wrong_count}")
    print(f"Папка с результатами: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()