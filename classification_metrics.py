"""Recalculate the fixed-split comparison between Logistic Regression and ClockwiseCNN.

The script trains Logistic Regression on `training/`, loads one saved CNN
checkpoint, and evaluates both models on training, validation, and testing
folders. It does not update CNN weights or create a new model checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms

from model import ClockwiseCNN
from train import (
    ClockwiseImageDataset,
    ImageSample,
    calculate_binary_metrics,
    choose_device,
    create_baseline_features,
    read_labelled_images,
    validate_split_independence,
)


METRIC_COLUMNS = ("accuracy", "precision", "recall", "f1", "roc_auc")


def create_checkpoint_transform(checkpoint: dict) -> transforms.Compose:
    """Use the image size and normalization values stored with the CNN."""
    image_size = checkpoint["image_size"]
    mean = checkpoint["mean"]
    standard_deviation = checkpoint["std"]
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((mean,), (standard_deviation,)),
        ]
    )


def evaluate_logistic_regression(
    training_samples: list[ImageSample],
    splits: dict[str, list[ImageSample]],
    seed: int,
) -> dict[str, dict[str, float]]:
    """Train the baseline on training pixels and evaluate every fixed split."""
    classifier = LogisticRegression(solver="liblinear", max_iter=1000, random_state=seed)
    training_features, training_labels = create_baseline_features(training_samples)
    classifier.fit(training_features, training_labels)

    metrics_by_split: dict[str, dict[str, float]] = {}
    for split_name, samples in splits.items():
        features, labels = create_baseline_features(samples)
        positive_probabilities = classifier.predict_proba(features)[:, 1]
        metrics_by_split[split_name] = calculate_binary_metrics(labels, positive_probabilities)
    return metrics_by_split


@torch.no_grad()
def evaluate_cnn(
    model: ClockwiseCNN,
    samples: list[ImageSample],
    transform: transforms.Compose,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate a frozen CNN checkpoint without calculating gradients."""
    loader = DataLoader(ClockwiseImageDataset(samples, transform), batch_size=batch_size)
    model.eval()
    all_logits: list[Tensor] = []
    all_labels: list[Tensor] = []

    for images, labels in loader:
        logits = model(images.to(device)).squeeze(1)
        all_logits.append(logits.cpu())
        all_labels.append(labels)

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy().astype(int)
    positive_probabilities = 1 / (1 + np.exp(-logits))
    return calculate_binary_metrics(labels, positive_probabilities)


def evaluate_cnn_checkpoint(
    checkpoint_path: Path,
    splits: dict[str, list[ImageSample]],
    batch_size: int,
) -> tuple[dict[str, dict[str, float]], dict, str]:
    """Load one saved CNN checkpoint and calculate metrics for all splits."""
    device = choose_device()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = ClockwiseCNN().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    transform = create_checkpoint_transform(checkpoint)

    metrics_by_split = {
        split_name: evaluate_cnn(model, samples, transform, batch_size, device)
        for split_name, samples in splits.items()
    }
    return metrics_by_split, checkpoint, str(device)


def print_comparison_table(models: dict[str, dict[str, dict[str, float]]]) -> None:
    """Print one reproducible table for the report."""
    header = f"{'Model':<22} {'Split':<12}" + "".join(f"{name:>11}" for name in METRIC_COLUMNS)
    print(header)
    print("-" * len(header))

    for model_name, metrics_by_split in models.items():
        for split_name in ("train", "validation", "test"):
            metrics = metrics_by_split[split_name]
            values = "".join(f"{metrics[name]:>11.3f}" for name in METRIC_COLUMNS)
            print(f"{model_name:<22} {split_name:<12}{values}")


def output_path_for(checkpoint_path: Path, output_path: Path | None) -> Path:
    """Use a checkpoint-specific JSON filename unless the user supplies one."""
    if output_path is not None:
        return output_path
    return checkpoint_path.parent / f"classification_metrics_{checkpoint_path.stem}.json"


def run(args: argparse.Namespace) -> None:
    """Generate and save a fresh model-comparison report from the fixed folders."""
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"CNN checkpoint was not found: {args.checkpoint}")

    splits = {
        "train": read_labelled_images(args.training_directory),
        "validation": read_labelled_images(args.validation_directory),
        "test": read_labelled_images(args.testing_directory),
    }
    validate_split_independence(splits)

    logistic_metrics = evaluate_logistic_regression(splits["train"], splits, args.seed)
    cnn_metrics, checkpoint, device = evaluate_cnn_checkpoint(
        args.checkpoint,
        splits,
        args.batch_size,
    )
    models = {
        "Logistic Regression": logistic_metrics,
        "ClockwiseCNN": cnn_metrics,
    }
    print_comparison_table(models)

    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_best_epoch": checkpoint.get("best_epoch"),
        "checkpoint_best_validation_loss": checkpoint.get("best_validation_loss"),
        "device": device,
        "split_sizes": {name: len(samples) for name, samples in splits.items()},
        "models": models,
    }
    output_path = output_path_for(args.checkpoint, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved reproducible metrics to {output_path}")


def parse_arguments() -> argparse.Namespace:
    """Read paths for the fixed splits and the CNN checkpoint being compared."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", dest="training_directory", type=Path, default=Path("training"))
    parser.add_argument("--validation-dir", dest="validation_directory", type=Path, default=Path("validation"))
    parser.add_argument("--testing-dir", dest="testing_directory", type=Path, default=Path("testing"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/clockwise_cnn.pt"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_arguments())
