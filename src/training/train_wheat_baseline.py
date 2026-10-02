"""Train and evaluate a manifest-driven wheat image-classification baseline.

This script deliberately consumes only ``results/data/clean_split_manifest.csv``
and never uses the original supplier split directories. Its small CNN is a
sanity-check baseline: it verifies that the data loader, preprocessing,
training loop, and evaluation outputs work before transfer-learning experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "results" / "data" / "clean_split_manifest.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "wheat_baseline"
IMAGE_SIZE = 224


@dataclass(frozen=True)
class Sample:
    path: Path
    label: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_samples(manifest: Path) -> tuple[dict[str, list[Sample]], list[str]]:
    by_split: dict[str, list[Sample]] = {"train": [], "validation": [], "test": []}
    with manifest.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            if row["task"] == "wheat_condition" and row["split"] in by_split:
                by_split[row["split"]].append(Sample(Path(row["path"]), row["label"]))
    labels = sorted({sample.label for samples in by_split.values() for sample in samples})
    if not labels or not all(by_split.values()):
        raise ValueError("Manifest has no complete wheat train/validation/test split.")
    return by_split, labels


class WheatDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, samples: list[Sample], label_to_index: dict[str, int], augment: bool) -> None:
        self.samples = samples
        self.label_to_index = label_to_index
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            image = image.convert("RGB")
            # Center-crop after resizing the shorter edge. This avoids black padding
            # becoming a spurious visual feature for the classifier.
            width, height = image.size
            scale = IMAGE_SIZE / min(width, height)
            image = image.resize((round(width * scale), round(height * scale)), Image.Resampling.BILINEAR)
            left = (image.width - IMAGE_SIZE) // 2
            top = (image.height - IMAGE_SIZE) // 2
            image = image.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE))
            if self.augment and random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            pixels = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(pixels).permute(2, 0, 1)
        # ImageNet-like normalization is not required for this from-scratch model,
        # but centering the input makes optimization more stable.
        tensor = (tensor - 0.5) / 0.5
        return tensor, self.label_to_index[sample.label]


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.25), nn.Linear(128, num_classes))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, list[int], list[int]]:
    model.eval()
    correct = total = 0
    actual: list[int] = []
    predicted: list[int] = []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.numel()
        actual.extend(labels.cpu().tolist())
        predicted.extend(predictions.cpu().tolist())
    return correct / total, actual, predicted


def save_confusion_matrix(matrix: np.ndarray, labels: list[str], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(13, 11))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(xticks=np.arange(len(labels)), yticks=np.arange(len(labels)),
             xticklabels=labels, yticklabels=labels, ylabel="True label", xlabel="Predicted label")
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center",
                      color="white" if matrix[row, column] > threshold else "black", fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")

    set_seed(args.seed)
    samples, labels = read_samples(args.manifest.resolve())
    label_to_index = {label: index for index, label in enumerate(labels)}
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(WheatDataset(samples["train"], label_to_index, augment=True), args.batch_size, shuffle=True)
    validation_loader = DataLoader(WheatDataset(samples["validation"], label_to_index, augment=False), args.batch_size)
    test_loader = DataLoader(WheatDataset(samples["test"], label_to_index, augment=False), args.batch_size)

    counts = np.array([sum(sample.label == label for sample in samples["train"]) for label in labels])
    weights = torch.tensor(counts.sum() / (len(labels) * counts), dtype=torch.float32, device=device)
    model = SmallCNN(len(labels)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(weight=weights)

    best_validation_f1 = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = total_examples = 0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * targets.numel()
            total_examples += targets.numel()
        validation_accuracy, actual, predicted = evaluate(model, validation_loader, device)
        validation_f1 = f1_score(actual, predicted, average="macro", zero_division=0)
        row = {"epoch": epoch, "train_loss": total_loss / total_examples,
               "validation_accuracy": validation_accuracy, "validation_macro_f1": validation_f1}
        history.append(row)
        print(f"epoch={epoch} loss={row['train_loss']:.4f} val_accuracy={validation_accuracy:.4f} val_macro_f1={validation_f1:.4f}")
        if validation_f1 > best_validation_f1:
            best_validation_f1 = validation_f1
            torch.save({"model_state": model.state_dict(), "labels": labels, "image_size": IMAGE_SIZE}, args.output_dir / "best_model.pt")

    checkpoint = torch.load(args.output_dir / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    test_accuracy, actual, predicted = evaluate(model, test_loader, device)
    matrix = confusion_matrix(actual, predicted, labels=list(range(len(labels))))
    save_confusion_matrix(matrix, labels, args.output_dir / "confusion_matrix.png")
    metrics = {
        "task": "wheat_condition", "architecture": "SmallCNN", "device": str(device), "seed": args.seed,
        "labels": labels, "test_accuracy": test_accuracy,
        "test_macro_f1": f1_score(actual, predicted, average="macro", zero_division=0),
        "classification_report": classification_report(actual, predicted, target_names=labels, output_dict=True, zero_division=0),
        "history": history,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"test_accuracy={metrics['test_accuracy']:.4f} test_macro_f1={metrics['test_macro_f1']:.4f}")
    print(f"outputs={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
