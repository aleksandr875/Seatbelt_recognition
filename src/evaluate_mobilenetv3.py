from pathlib import Path
import json
import time
import csv

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_PATH = PROJECT_ROOT / "models" / "best_mobilenetv3.pt"
RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 16
IMAGE_SIZE = 224


class ImageFolderWithPaths(datasets.ImageFolder):
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
    model = models.mobilenet_v3_small(weights=None)

    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    return model


def save_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(7, 6))

    im = ax.imshow(cm)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Предсказанный класс",
        ylabel="Истинный класс",
        title="Confusion matrix"
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    fig.tight_layout()

    output_path = RESULTS_DIR / "confusion_matrix.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Матрица ошибок сохранена: {output_path}")


def save_wrong_predictions(wrong_predictions):
    output_path = RESULTS_DIR / "wrong_predictions.csv"

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "file_path",
                "true_class",
                "predicted_class",
                "confidence"
            ]
        )

        writer.writeheader()

        for item in wrong_predictions:
            writer.writerow(item)

    print(f"Ошибочные предсказания сохранены: {output_path}")


def evaluate():
    device = get_device()
    print(f"Используемое устройство: {device}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Модель не найдена: {MODEL_PATH}")

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
    class_to_idx = test_dataset.class_to_idx

    print("Классы:")
    print(class_to_idx)

    model = create_model(
        num_classes=len(class_names),
        device=device
    )

    all_labels = []
    all_preds = []
    all_probs = []
    wrong_predictions = []

    total_inference_time = 0.0
    total_images = 0

    with torch.no_grad():
        for images, labels, paths in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            start_time = time.time()

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, preds = torch.max(probabilities, 1)

            end_time = time.time()

            total_inference_time += end_time - start_time
            total_images += images.size(0)

            labels_cpu = labels.cpu().numpy()
            preds_cpu = preds.cpu().numpy()
            probs_cpu = probabilities.cpu().numpy()

            all_labels.extend(labels_cpu)
            all_preds.extend(preds_cpu)
            all_probs.extend(probs_cpu)

            for true_label, pred_label, prob, path in zip(
                labels_cpu,
                preds_cpu,
                probs_cpu,
                paths
            ):
                if true_label != pred_label:
                    wrong_predictions.append({
                        "file_path": path,
                        "true_class": class_names[true_label],
                        "predicted_class": class_names[pred_label],
                        "confidence": float(prob[pred_label])
                    })

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    accuracy = accuracy_score(all_labels, all_preds)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)

    report_dict = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        digits=4,
        output_dict=True
    )

    report_text = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        digits=4
    )

    cm = confusion_matrix(all_labels, all_preds)

    avg_time_per_image = total_inference_time / total_images
    fps = total_images / total_inference_time

    metrics = {
        "classes": class_names,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "avg_time_per_image_seconds": avg_time_per_image,
        "fps": fps,
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist()
    }

    try:
        if len(class_names) == 2:
            positive_class_idx = 0
            y_true_binary = (all_labels == positive_class_idx).astype(int)
            y_score_binary = all_probs[:, positive_class_idx]
            metrics["roc_auc"] = roc_auc_score(y_true_binary, y_score_binary)
        else:
            metrics["roc_auc_macro_ovr"] = roc_auc_score(
                all_labels,
                all_probs,
                multi_class="ovr",
                average="macro"
            )
    except ValueError:
        metrics["roc_auc"] = None

    print()
    print("=== Основные метрики ===")
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Balanced accuracy: {balanced_acc:.4f}")
    print(f"Среднее время на 1 изображение: {avg_time_per_image:.4f} сек.")
    print(f"FPS: {fps:.2f}")

    if "roc_auc_macro_ovr" in metrics:
        print(f"ROC-AUC macro OVR: {metrics['roc_auc_macro_ovr']:.4f}")

    if "roc_auc" in metrics:
        print(f"ROC-AUC: {metrics['roc_auc']}")

    print()
    print("=== Classification report ===")
    print(report_text)

    print("=== Confusion matrix ===")
    print(cm)

    metrics_path = RESULTS_DIR / "evaluation_metrics.json"

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=4)

    print(f"\nМетрики сохранены: {metrics_path}")

    report_path = RESULTS_DIR / "classification_report.txt"

    with report_path.open("w", encoding="utf-8") as file:
        file.write(report_text)

    print(f"Classification report сохранен: {report_path}")

    save_confusion_matrix(cm, class_names)
    save_wrong_predictions(wrong_predictions)

    print(f"\nКоличество ошибочных предсказаний: {len(wrong_predictions)}")


if __name__ == "__main__":
    evaluate()