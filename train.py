"""Train a Logistic Regression baseline and a CNN on the labelled figures."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from model import ClockwiseCNN


FILENAME_PATTERN = re.compile(r"^\d+_([+-])\.png$")
IMAGE_SIZE = 128
BASELINE_SIZE = 64


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int


class FigureDataset(Dataset):
    def __init__(self, samples: list[Sample], transform) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            tensor = self.transform(image.convert("L"))
        return tensor, torch.tensor(float(sample.label), dtype=torch.float32)


def collect_samples(folder: Path) -> list[Sample]:
    samples = []
    for path in sorted(folder.glob("*.png")):
        match = FILENAME_PATTERN.match(path.name)
        if match:
            samples.append(Sample(path, int(match.group(1) == "+")))
    if not samples:
        raise ValueError(f"Expected labelled PNGs in {folder}")
    return samples


def validate_splits(splits: dict[str, list[Sample]]) -> None:
    """Reject a split when the same image content appears in more than one folder."""
    seen_hashes: dict[str, str] = {}
    for split_name, samples in splits.items():
        for sample in samples:
            image_hash = hashlib.sha256(sample.path.read_bytes()).hexdigest()
            previous_split = seen_hashes.get(image_hash)
            if previous_split and previous_split != split_name:
                raise ValueError(f"Duplicate image found in {previous_split} and {split_name}: {sample.path.name}")
            seen_hashes[image_hash] = split_name


def train_normalization(samples: list[Sample]) -> tuple[float, float]:
    """Fit normalization values from training images only to avoid test-set leakage."""
    pixel_sum = 0.0
    squared_sum = 0.0
    total_pixels = 0
    for sample in samples:
        with Image.open(sample.path) as image:
            pixels = np.asarray(image.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE)), dtype=np.float32) / 255.0
        pixel_sum += float(pixels.sum())
        squared_sum += float((pixels * pixels).sum())
        total_pixels += pixels.size
    mean = pixel_sum / total_pixels
    variance = max(squared_sum / total_pixels - mean * mean, 1e-8)
    return mean, variance**0.5


def transforms_for(mean: float, std: float):
    normalization = transforms.Normalize((mean,), (std,))
    train_transform = transforms.Compose(
        [
            # Reflections invert clockwise order, so they are intentionally excluded.
            transforms.RandomAffine(degrees=8, translate=(0.04, 0.04), scale=(0.95, 1.05), fill=255),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            normalization,
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            normalization,
        ]
    )
    return train_transform, evaluation_transform


def raw_features(samples: list[Sample]) -> tuple[np.ndarray, np.ndarray]:
    features = []
    labels = []
    for sample in samples:
        with Image.open(sample.path) as image:
            pixels = np.asarray(image.convert("L").resize((BASELINE_SIZE, BASELINE_SIZE)), dtype=np.float32) / 255.0
        features.append(pixels.ravel())
        labels.append(sample.label)
    return np.stack(features), np.asarray(labels)


def metric_values(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = (probabilities >= 0.5).astype(int)
    true_positive = int(np.sum((labels == 1) & (predictions == 1)))
    false_positive = int(np.sum((labels == 0) & (predictions == 1)))
    false_negative = int(np.sum((labels == 1) & (predictions == 0)))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": float(roc_auc_score(labels, probabilities)),
    }


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    loss_total = 0.0
    sample_count = 0
    logits_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    for images, labels in loader:
        labels = labels.to(device)
        logits = model(images.to(device)).squeeze(1)
        loss_total += float(criterion(logits, labels).item()) * len(labels)
        sample_count += len(labels)
        logits_list.append(logits.cpu())
        labels_list.append(labels.cpu())
    logits = torch.cat(logits_list).numpy()
    labels = torch.cat(labels_list).numpy().astype(int)
    probabilities = 1 / (1 + np.exp(-logits))
    return loss_total / sample_count, metric_values(labels, probabilities), labels, probabilities


def save_training_curves(history: list[dict[str, float]], output: Path) -> None:
    epochs = range(1, len(history) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["validation_loss"] for row in history], label="validation")
    axes[0].set(title="BCE loss", xlabel="epoch", ylabel="loss")
    axes[0].legend()
    axes[1].plot(epochs, [row["train_f1"] for row in history], label="train")
    axes[1].plot(epochs, [row["validation_f1"] for row in history], label="validation")
    axes[1].set(title="F1 score", xlabel="epoch", ylabel="F1")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def save_confusion_matrix(labels: np.ndarray, probabilities: np.ndarray, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(labels, probabilities >= 0.5, display_labels=["- (ACB)", "+ (ABC)"], cmap="Blues", ax=axis)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def run(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_samples = collect_samples(args.training_dir)
    validation_samples = collect_samples(args.validation_dir)
    test_samples = collect_samples(args.testing_dir)
    validate_splits({"training": train_samples, "validation": validation_samples, "testing": test_samples})
    mean, std = train_normalization(train_samples)
    train_transform, evaluation_transform = transforms_for(mean, std)
    loaders = {
        "train": DataLoader(FigureDataset(train_samples, train_transform), batch_size=args.batch_size, shuffle=True),
        "train_eval": DataLoader(FigureDataset(train_samples, evaluation_transform), batch_size=args.batch_size),
        "validation": DataLoader(FigureDataset(validation_samples, evaluation_transform), batch_size=args.batch_size),
        "test": DataLoader(FigureDataset(test_samples, evaluation_transform), batch_size=args.batch_size),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)

    baseline = LogisticRegression(solver="liblinear", max_iter=1000, random_state=args.seed)
    baseline_x_train, baseline_y_train = raw_features(train_samples)
    baseline.fit(baseline_x_train, baseline_y_train)
    baseline_metrics = {}
    for name, split in {"train": train_samples, "validation": validation_samples, "test": test_samples}.items():
        features, labels = raw_features(split)
        baseline_metrics[name] = metric_values(labels, baseline.predict_proba(features)[:, 1])

    device = choose_device()
    model = ClockwiseCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    history = []
    best_validation_loss = float("inf")
    best_epoch = 0
    patience_left = args.patience
    checkpoint_path = args.output_dir / "clockwise_cnn.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        for images, labels in loaders["train"]:
            optimizer.zero_grad()
            logits = model(images.to(device)).squeeze(1)
            loss = criterion(logits, labels.to(device))
            loss.backward()
            optimizer.step()
        train_loss, train_metrics, _, _ = evaluate(model, loaders["train_eval"], device)
        validation_loss, validation_metrics, _, _ = evaluate(model, loaders["validation"], device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "train_f1": train_metrics["f1"],
                "validation_f1": validation_metrics["f1"],
            }
        )
        print(f"Epoch {epoch:03d}  train_loss={train_loss:.4f}  val_loss={validation_loss:.4f}  val_f1={validation_metrics['f1']:.3f}")
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            patience_left = args.patience
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "mean": mean,
                    "std": std,
                    "image_size": IMAGE_SIZE,
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_validation_loss,
                },
                checkpoint_path,
            )
        else:
            patience_left -= 1
            if patience_left == 0:
                print("Early stopping selected the best validation-loss checkpoint.")
                break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    _, train_values, _, _ = evaluate(model, loaders["train_eval"], device)
    _, validation_values, _, _ = evaluate(model, loaders["validation"], device)
    _, test_values, test_labels, test_probabilities = evaluate(model, loaders["test"], device)
    cnn_metrics = {
        "train": train_values,
        "validation": validation_values,
        "test": test_values,
    }

    save_training_curves(history, args.output_dir / "training_curves.png")
    save_confusion_matrix(test_labels, test_probabilities, args.output_dir / "test_confusion_matrix.png")
    report = {
        "seed": args.seed,
        "device": str(device),
        "split": {
            "train": len(train_samples),
            "validation": len(validation_samples),
            "test": len(test_samples),
            "train_positive": sum(sample.label for sample in train_samples),
            "validation_positive": sum(sample.label for sample in validation_samples),
            "test_positive": sum(sample.label for sample in test_samples),
        },
        "normalization": {"mean": mean, "std": std},
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "baseline_logistic_regression": baseline_metrics,
        "cnn": cnn_metrics,
        "epochs_ran": len(history),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, default=Path("training"))
    parser.add_argument("--validation-dir", type=Path, default=Path("validation"))
    parser.add_argument("--testing-dir", type=Path, default=Path("testing"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(arguments())
