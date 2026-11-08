"""Train and evaluate the clockwise-order image classifiers.

The filename suffix supplies each supervised target. In `labels.py`, `ABC` is
defined as class `+` (target 1) and `ACB` as class `-` (target 0). The CNN sees
only image pixels. It learns the visual relationship between those pixels and
the targets during training.
"""

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
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from labels import class_id_from_symbol, symbol_from_filename
from model import ClockwiseCNN


IMAGE_SIZE = 128
BASELINE_IMAGE_SIZE = 64
MODEL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


@dataclass(frozen=True)
class ImageSample:
    """One labelled PNG image and its binary training target."""

    path: Path
    class_id: int


@dataclass(frozen=True)
class EvaluationResult:
    """Metrics and raw outputs from evaluating a CNN on one dataset split."""

    loss: float
    metrics: dict[str, float]
    labels: np.ndarray
    probabilities: np.ndarray


@dataclass(frozen=True)
class TrainingResult:
    """Information needed to evaluate and document the selected checkpoint."""

    history: list[dict[str, float]]
    best_epoch: int
    best_validation_loss: float


class ClockwiseImageDataset(Dataset):
    """Load labelled grayscale images and apply one supplied preprocessing flow."""

    def __init__(self, samples: list[ImageSample], transform) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            grayscale_image = image.convert("L")
            image_tensor = self.transform(grayscale_image)
        target = torch.tensor(float(sample.class_id), dtype=torch.float32)
        return image_tensor, target


def image_id(path: Path) -> int:
    """Read the numeric image identifier from a filename such as `170_+.png`."""
    return int(path.name.split("_", maxsplit=1)[0])


def read_labelled_images(directory: Path) -> list[ImageSample]:
    """Read the labelled PNG files in one fixed dataset folder."""
    samples: list[ImageSample] = []
    for image_path in sorted(directory.glob("*.png"), key=image_id):
        symbol = symbol_from_filename(image_path.name)
        if symbol is None:
            continue
        samples.append(ImageSample(image_path, class_id_from_symbol(symbol)))

    if not samples:
        raise ValueError(f"Expected labelled PNG images in {directory}")
    return samples


def validate_model_name(model_name: str) -> None:
    """Reject model names that could create unexpected artifact paths."""
    if not MODEL_NAME_PATTERN.fullmatch(model_name):
        message = "Model name may contain only letters, numbers, underscores, and hyphens."
        raise ValueError(message)


def validate_split_independence(splits: dict[str, list[ImageSample]]) -> None:
    """Stop training if identical image content appears in two different splits."""
    split_by_image_hash: dict[str, str] = {}

    for split_name, samples in splits.items():
        for sample in samples:
            image_hash = hashlib.sha256(sample.path.read_bytes()).hexdigest()
            previous_split = split_by_image_hash.get(image_hash)
            if previous_split is not None and previous_split != split_name:
                message = f"Duplicate image found in {previous_split} and {split_name}: {sample.path.name}"
                raise ValueError(message)
            split_by_image_hash[image_hash] = split_name


def calculate_training_normalization(samples: list[ImageSample]) -> tuple[float, float]:
    """Calculate grayscale normalization values from training images only."""
    pixel_sum = 0.0
    squared_pixel_sum = 0.0
    pixel_count = 0

    for sample in samples:
        with Image.open(sample.path) as image:
            pixels = np.asarray(
                image.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE)),
                dtype=np.float32,
            ) / 255.0
        pixel_sum += float(pixels.sum())
        squared_pixel_sum += float((pixels * pixels).sum())
        pixel_count += pixels.size

    mean = pixel_sum / pixel_count
    variance = max(squared_pixel_sum / pixel_count - mean * mean, 1e-8)
    return mean, variance**0.5


def create_transforms(mean: float, standard_deviation: float):
    """Create separate image pipelines for training and evaluation."""
    normalize = transforms.Normalize((mean,), (standard_deviation,))

    training_transform = transforms.Compose(
        [
            # Reflections reverse clockwise order, so flips are intentionally excluded.
            transforms.RandomAffine(
                degrees=8,
                translate=(0.04, 0.04),
                scale=(0.95, 1.05),
                fill=255,
            ),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return training_transform, evaluation_transform


def create_baseline_features(samples: list[ImageSample]) -> tuple[np.ndarray, np.ndarray]:
    """Flatten 64 by 64 grayscale images for the Logistic Regression baseline."""
    feature_rows: list[np.ndarray] = []
    target_values: list[int] = []

    for sample in samples:
        with Image.open(sample.path) as image:
            pixels = np.asarray(
                image.convert("L").resize((BASELINE_IMAGE_SIZE, BASELINE_IMAGE_SIZE)),
                dtype=np.float32,
            ) / 255.0
        feature_rows.append(pixels.ravel())
        target_values.append(sample.class_id)

    return np.stack(feature_rows), np.asarray(target_values)


def calculate_binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    """Calculate the required metrics for positive class `+` at a 0.5 threshold."""
    predictions = (probabilities >= 0.5).astype(int)

    true_positives = int(np.sum((labels == 1) & (predictions == 1)))
    false_positives = int(np.sum((labels == 0) & (predictions == 1)))
    false_negatives = int(np.sum((labels == 1) & (predictions == 0)))

    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives
    precision = true_positives / precision_denominator if precision_denominator else 0.0
    recall = true_positives / recall_denominator if recall_denominator else 0.0
    f1_denominator = precision + recall
    f1_score = 2 * precision * recall / f1_denominator if f1_denominator else 0.0

    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": precision,
        "recall": recall,
        "f1": f1_score,
        "roc_auc": float(roc_auc_score(labels, probabilities)),
    }


def choose_device() -> torch.device:
    """Use CUDA or Apple MPS when available, otherwise use the CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate_cnn(model: nn.Module, loader: DataLoader, device: torch.device) -> EvaluationResult:
    """Evaluate a CNN without changing its learned weights."""
    model.eval()
    loss_function = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_images = 0
    logits_by_batch: list[Tensor] = []
    labels_by_batch: list[Tensor] = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images).squeeze(1)

        batch_loss = loss_function(logits, labels)
        total_loss += float(batch_loss.item()) * len(labels)
        total_images += len(labels)
        logits_by_batch.append(logits.cpu())
        labels_by_batch.append(labels.cpu())

    logits = torch.cat(logits_by_batch).numpy()
    labels = torch.cat(labels_by_batch).numpy().astype(int)
    probabilities = 1 / (1 + np.exp(-logits))
    metrics = calculate_binary_metrics(labels, probabilities)
    return EvaluationResult(total_loss / total_images, metrics, labels, probabilities)


def train_one_epoch(
    model: ClockwiseCNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.BCEWithLogitsLoss,
    device: torch.device,
) -> None:
    """Update the CNN weights once using every image in the training loader."""
    model.train()
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images).squeeze(1)
        loss = loss_function(logits, labels)
        loss.backward()
        optimizer.step()


def train_cnn(
    model: ClockwiseCNN,
    train_loader: DataLoader,
    train_evaluation_loader: DataLoader,
    validation_loader: DataLoader,
    checkpoint_path: Path,
    mean: float,
    standard_deviation: float,
    device: torch.device,
    maximum_epochs: int,
    learning_rate: float,
    patience: int,
) -> TrainingResult:
    """Train the CNN and save the checkpoint with the lowest validation loss."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_function = nn.BCEWithLogitsLoss()
    history: list[dict[str, float]] = []
    best_validation_loss = float("inf")
    best_epoch = 0
    remaining_patience = patience

    for epoch in range(1, maximum_epochs + 1):
        train_one_epoch(model, train_loader, optimizer, loss_function, device)
        train_evaluation = evaluate_cnn(model, train_evaluation_loader, device)
        validation_evaluation = evaluate_cnn(model, validation_loader, device)

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_evaluation.loss,
                "validation_loss": validation_evaluation.loss,
                "train_f1": train_evaluation.metrics["f1"],
                "validation_f1": validation_evaluation.metrics["f1"],
            }
        )
        print(
            f"Epoch {epoch:03d}  "
            f"train_loss={train_evaluation.loss:.4f}  "
            f"validation_loss={validation_evaluation.loss:.4f}  "
            f"validation_f1={validation_evaluation.metrics['f1']:.3f}"
        )

        if validation_evaluation.loss < best_validation_loss:
            best_validation_loss = validation_evaluation.loss
            best_epoch = epoch
            remaining_patience = patience
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "mean": mean,
                    "std": standard_deviation,
                    "image_size": IMAGE_SIZE,
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_validation_loss,
                },
                checkpoint_path,
            )
            continue

        remaining_patience -= 1
        if remaining_patience == 0:
            print("Early stopping selected the checkpoint with the lowest validation loss.")
            break

    return TrainingResult(history, best_epoch, best_validation_loss)


def evaluate_logistic_regression(
    training_samples: list[ImageSample],
    validation_samples: list[ImageSample],
    testing_samples: list[ImageSample],
    seed: int,
) -> dict[str, dict[str, float]]:
    """Fit the raw-pixel baseline and report metrics for every fixed split."""
    classifier = LogisticRegression(solver="liblinear", max_iter=1000, random_state=seed)
    training_features, training_labels = create_baseline_features(training_samples)
    classifier.fit(training_features, training_labels)

    metrics_by_split: dict[str, dict[str, float]] = {}
    splits = {
        "train": training_samples,
        "validation": validation_samples,
        "test": testing_samples,
    }
    for split_name, samples in splits.items():
        features, labels = create_baseline_features(samples)
        positive_probabilities = classifier.predict_proba(features)[:, 1]
        metrics_by_split[split_name] = calculate_binary_metrics(labels, positive_probabilities)
    return metrics_by_split


def save_training_curves(history: list[dict[str, float]], output_path: Path) -> None:
    """Save loss and F1 curves for the selected CNN training run."""
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
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_confusion_matrix(labels: np.ndarray, probabilities: np.ndarray, output_path: Path) -> None:
    """Save the final test confusion matrix using the required class names."""
    figure, axis = plt.subplots(figsize=(4, 4))
    predictions = probabilities >= 0.5
    ConfusionMatrixDisplay.from_predictions(
        labels,
        predictions,
        display_labels=["- (ACB)", "+ (ABC)"],
        cmap="Blues",
        ax=axis,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def set_random_seed(seed: int) -> None:
    """Make model initialization and training augmentation reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_training(args: argparse.Namespace) -> None:
    """Run the full fixed-split experiment and write named artifacts."""
    validate_model_name(args.model_name)
    set_random_seed(args.seed)

    training_samples = read_labelled_images(args.training_directory)
    validation_samples = read_labelled_images(args.validation_directory)
    testing_samples = read_labelled_images(args.testing_directory)
    validate_split_independence(
        {
            "training": training_samples,
            "validation": validation_samples,
            "testing": testing_samples,
        }
    )

    mean, standard_deviation = calculate_training_normalization(training_samples)
    training_transform, evaluation_transform = create_transforms(mean, standard_deviation)
    loaders = {
        "training": DataLoader(
            ClockwiseImageDataset(training_samples, training_transform),
            batch_size=args.batch_size,
            shuffle=True,
        ),
        "training_evaluation": DataLoader(
            ClockwiseImageDataset(training_samples, evaluation_transform),
            batch_size=args.batch_size,
        ),
        "validation": DataLoader(
            ClockwiseImageDataset(validation_samples, evaluation_transform),
            batch_size=args.batch_size,
        ),
        "testing": DataLoader(
            ClockwiseImageDataset(testing_samples, evaluation_transform),
            batch_size=args.batch_size,
        ),
    }

    args.output_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_directory / f"{args.model_name}.pt"
    baseline_metrics = evaluate_logistic_regression(
        training_samples,
        validation_samples,
        testing_samples,
        args.seed,
    )

    device = choose_device()
    cnn = ClockwiseCNN().to(device)
    training_result = train_cnn(
        model=cnn,
        train_loader=loaders["training"],
        train_evaluation_loader=loaders["training_evaluation"],
        validation_loader=loaders["validation"],
        checkpoint_path=checkpoint_path,
        mean=mean,
        standard_deviation=standard_deviation,
        device=device,
        maximum_epochs=args.epochs,
        learning_rate=args.learning_rate,
        patience=args.patience,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    cnn.load_state_dict(checkpoint["state_dict"])
    training_evaluation = evaluate_cnn(cnn, loaders["training_evaluation"], device)
    validation_evaluation = evaluate_cnn(cnn, loaders["validation"], device)
    testing_evaluation = evaluate_cnn(cnn, loaders["testing"], device)

    save_training_curves(
        training_result.history,
        args.output_directory / f"{args.model_name}_training_curves.png",
    )
    save_confusion_matrix(
        testing_evaluation.labels,
        testing_evaluation.probabilities,
        args.output_directory / f"{args.model_name}_test_confusion_matrix.png",
    )

    report = {
        "model_name": args.model_name,
        "seed": args.seed,
        "device": str(device),
        "split": {
            "train": len(training_samples),
            "validation": len(validation_samples),
            "test": len(testing_samples),
            "train_positive": sum(sample.class_id for sample in training_samples),
            "validation_positive": sum(sample.class_id for sample in validation_samples),
            "test_positive": sum(sample.class_id for sample in testing_samples),
        },
        "normalization": {"mean": mean, "std": standard_deviation},
        "best_epoch": training_result.best_epoch,
        "best_validation_loss": training_result.best_validation_loss,
        "baseline_logistic_regression": baseline_metrics,
        "cnn": {
            "train": training_evaluation.metrics,
            "validation": validation_evaluation.metrics,
            "test": testing_evaluation.metrics,
        },
        "epochs_ran": len(training_result.history),
    }
    metrics_path = args.output_directory / f"{args.model_name}_metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def parse_arguments() -> argparse.Namespace:
    """Read command-line options while keeping the fixed-split defaults visible."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", dest="training_directory", type=Path, default=Path("training"))
    parser.add_argument("--validation-dir", dest="validation_directory", type=Path, default=Path("validation"))
    parser.add_argument("--testing-dir", dest="testing_directory", type=Path, default=Path("testing"))
    parser.add_argument("--output-dir", dest="output_directory", type=Path, default=Path("artifacts"))
    parser.add_argument("--model-name", default="clockwise_cnn")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_arguments())
