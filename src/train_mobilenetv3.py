from pathlib import Path
import copy
import time

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

MODELS_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 0.0001
IMAGE_SIZE = 224


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform, eval_transform


def get_dataloaders():
    train_transform, eval_transform = get_transforms()

    train_dataset = datasets.ImageFolder(
        DATA_DIR / "train",
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        DATA_DIR / "val",
        transform=eval_transform
    )

    test_dataset = datasets.ImageFolder(
        DATA_DIR / "test",
        transform=eval_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


def calculate_class_weights(train_dataset, device):
    targets = train_dataset.targets
    class_counts = torch.bincount(torch.tensor(targets))
    total_count = len(targets)

    class_weights = total_count / (len(class_counts) * class_counts.float())
    class_weights = class_weights.to(device)

    print("Количество изображений по классам в train:")
    for class_name, class_index in train_dataset.class_to_idx.items():
        print(f"{class_name}: {class_counts[class_index].item()}")

    print("Веса классов:")
    print(class_weights)

    return class_weights


def create_model(num_classes, device):
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)

    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)

    model = model.to(device)

    return model


def train_model(model, train_loader, val_loader, criterion, optimizer, device):
    best_model_weights = copy.deepcopy(model.state_dict())
    best_val_accuracy = 0.0

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()

        print(f"\nЭпоха {epoch + 1}/{NUM_EPOCHS}")

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
                data_loader = train_loader
            else:
                model.eval()
                data_loader = val_loader

            running_loss = 0.0
            running_corrects = 0
            total_samples = 0

            for inputs, labels in data_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                total_samples += inputs.size(0)

            epoch_loss = running_loss / total_samples
            epoch_accuracy = running_corrects / total_samples

            print(f"{phase}: loss={epoch_loss:.4f}, accuracy={epoch_accuracy:.4f}")

            if phase == "val" and epoch_accuracy > best_val_accuracy:
                best_val_accuracy = epoch_accuracy
                best_model_weights = copy.deepcopy(model.state_dict())

        elapsed_time = time.time() - start_time
        print(f"Время эпохи: {elapsed_time:.1f} сек.")

    print(f"\nЛучшая accuracy на val: {best_val_accuracy:.4f}")

    model.load_state_dict(best_model_weights)

    return model


def evaluate_model(model, test_loader, test_dataset, device):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    class_names = test_dataset.classes

    print("\nClassification report:")
    print(classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        digits=4
    ))

    print("Confusion matrix:")
    print(confusion_matrix(all_labels, all_preds))


def main():
    device = get_device()
    print(f"Используемое устройство: {device}")

    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = get_dataloaders()

    print("Классы:")
    print(train_dataset.class_to_idx)

    num_classes = len(train_dataset.classes)

    class_weights = calculate_class_weights(train_dataset, device)

    model = create_model(num_classes, device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device
    )

    model_path = MODELS_DIR / "best_mobilenetv3.pt"
    torch.save(model.state_dict(), model_path)

    print(f"\nМодель сохранена: {model_path}")

    evaluate_model(
        model=model,
        test_loader=test_loader,
        test_dataset=test_dataset,
        device=device
    )


if __name__ == "__main__":
    main()