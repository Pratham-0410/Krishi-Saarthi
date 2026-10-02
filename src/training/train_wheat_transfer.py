"""Train a frozen-backbone transfer-learning model on the clean wheat manifest.

Supported architectures are ``resnet18`` and ``mobilenet_v3_small``. Both use
ImageNet weights, train only their classifier head, select the best checkpoint
by validation macro F1, and evaluate the held-out test set exactly once.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "results" / "data" / "clean_split_manifest.csv"
IMAGE_SIZE = 224
NORMALIZE = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

# Keep downloaded ImageNet weights project-local rather than writing to a user-wide
# cache. Set before model construction so torchvision uses this location.
os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / ".cache" / "torch"))


@dataclass(frozen=True)
class Sample:
    path: Path
    label: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_samples(manifest: Path) -> tuple[dict[str, list[Sample]], list[str]]:
    samples: dict[str, list[Sample]] = {"train": [], "validation": [], "test": []}
    with manifest.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            if row["task"] == "wheat_condition" and row["split"] in samples:
                samples[row["split"]].append(Sample(Path(row["path"]), row["label"]))
    labels = sorted({sample.label for values in samples.values() for sample in values})
    if not labels or not all(samples.values()):
        raise ValueError("Manifest must contain non-empty wheat train, validation, and test splits.")
    return samples, labels


class ManifestDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, samples: list[Sample], label_to_index: dict[str, int], training: bool) -> None:
        self.samples = samples
        self.label_to_index = label_to_index
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.80, 1.0)) if training else transforms.Resize(256),
            transforms.RandomHorizontalFlip() if training else transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(), NORMALIZE,
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, self.label_to_index[sample.label]


def build_model(architecture: str, classes: int, pretrained: bool) -> nn.Module:
    if architecture == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, classes)
        return model
    if architecture == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, classes)
        return model
    raise ValueError(f"Unsupported architecture: {architecture}")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, list[int], list[int]]:
    model.eval()
    correct = total = 0
    truth: list[int] = []
    predictions: list[int] = []
    for images, targets in loader:
        outputs = model(images.to(device))
        guessed = outputs.argmax(dim=1).cpu()
        correct += (guessed == targets).sum().item()
        total += len(targets)
        truth.extend(targets.tolist())
        predictions.extend(guessed.tolist())
    return correct / total, truth, predictions


def save_confusion_matrix(matrix: np.ndarray, labels: list[str], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(13, 11))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(xticks=np.arange(len(labels)), yticks=np.arange(len(labels)), ylabel="True label", xlabel="Predicted label")
    axis.set_xticklabels(labels, rotation=45, ha="right")
    axis.set_yticklabels(labels)
    threshold = matrix.max() / 2 if matrix.size else 0
    for row, column in np.ndindex(matrix.shape):
        axis.text(column, row, matrix[row, column], ha="center", va="center", fontsize=8,
                  color="white" if matrix[row, column] > threshold else "black")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=("resnet18", "mobilenet_v3_small"), required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true", help="Use only for pipeline debugging; not a valid transfer-learning result.")
    parser.add_argument("--evaluate-only", action="store_true", help="Evaluate an existing best_model.pt without further training.")
    parser.add_argument("--completed-epochs", type=int, default=0, help="Epoch count to record when using --evaluate-only.")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")

    output_dir = args.output_dir or PROJECT_ROOT / "results" / f"wheat_{args.architecture}"
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    samples, labels = read_samples(args.manifest.resolve())
    label_to_index = {label: index for index, label in enumerate(labels)}
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    train_loader = DataLoader(ManifestDataset(samples["train"], label_to_index, training=True), args.batch_size, shuffle=True)
    validation_loader = DataLoader(ManifestDataset(samples["validation"], label_to_index, training=False), args.batch_size)
    test_loader = DataLoader(ManifestDataset(samples["test"], label_to_index, training=False), args.batch_size)

    model = build_model(args.architecture, len(labels), pretrained=not args.no_pretrained).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    counts = Counter(sample.label for sample in samples["train"])
    class_weights = torch.tensor([len(samples["train"]) / (len(labels) * counts[label]) for label in labels], device=device)
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    history = []
    if args.evaluate_only:
        checkpoint_path = output_dir / "best_model.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint to evaluate at {checkpoint_path}")
    else:
        best_f1 = -1.0
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = total = 0
            for images, targets in train_loader:
                images, targets = images.to(device), targets.to(device)
                optimizer.zero_grad()
                loss = criterion(model(images), targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(targets)
                total += len(targets)
            validation_accuracy, actual, predicted = evaluate(model, validation_loader, device)
            validation_f1 = f1_score(actual, predicted, average="macro", zero_division=0)
            result = {"epoch": epoch, "train_loss": total_loss / total, "validation_accuracy": validation_accuracy,
                      "validation_macro_f1": validation_f1}
            history.append(result)
            print(f"epoch={epoch} loss={result['train_loss']:.4f} val_accuracy={validation_accuracy:.4f} val_macro_f1={validation_f1:.4f}", flush=True)
            if validation_f1 > best_f1:
                best_f1 = validation_f1
                torch.save({"architecture": args.architecture, "model_state": model.state_dict(), "labels": labels,
                            "image_size": IMAGE_SIZE, "pretrained": not args.no_pretrained}, output_dir / "best_model.pt")

    checkpoint = torch.load(output_dir / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    test_accuracy, actual, predicted = evaluate(model, test_loader, device)
    matrix = confusion_matrix(actual, predicted, labels=list(range(len(labels))))
    save_confusion_matrix(matrix, labels, output_dir / "confusion_matrix.png")
    metrics = {
        "task": "wheat_condition", "architecture": args.architecture, "pretrained": not args.no_pretrained,
        "device": str(device), "seed": args.seed,
        "epochs": args.completed_epochs if args.evaluate_only else len(history), "labels": labels,
        "test_accuracy": test_accuracy, "test_macro_f1": f1_score(actual, predicted, average="macro", zero_division=0),
        "classification_report": classification_report(actual, predicted, target_names=labels, output_dict=True, zero_division=0),
        "history": history,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"test_accuracy={metrics['test_accuracy']:.4f} test_macro_f1={metrics['test_macro_f1']:.4f}")
    print(f"outputs={output_dir.resolve()}")


if __name__ == "__main__":
    main()
