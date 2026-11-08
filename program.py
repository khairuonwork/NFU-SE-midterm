"""Command-line gateway for training a named model or testing an existing model."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from labels import symbol_from_filename


PROJECT_DIRECTORY = Path(__file__).resolve().parent
ARTIFACTS_DIRECTORY = PROJECT_DIRECTORY / "artifacts"
TESTING_DIRECTORY = PROJECT_DIRECTORY / "testing"
MODEL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def available_models() -> list[Path]:
    return sorted(ARTIFACTS_DIRECTORY.glob("*.pt"))


def choose_model() -> Path | None:
    models = available_models()
    if not models:
        print("No trained model exists in artifacts/. Choose Train first.")
        return None

    print("\nAvailable models:")
    for number, model_path in enumerate(models, start=1):
        print(f"{number}. {model_path.stem}")

    selected = input("Choose model number: ").strip()
    if not selected.isdigit() or not 1 <= int(selected) <= len(models):
        print("Invalid model number.")
        return None
    return models[int(selected) - 1]


def metrics_path_for(model_path: Path) -> Path | None:
    named_metrics_path = ARTIFACTS_DIRECTORY / f"{model_path.stem}_metrics.json"
    if named_metrics_path.is_file():
        return named_metrics_path
    legacy_metrics_path = ARTIFACTS_DIRECTORY / "metrics.json"
    if model_path.stem == "clockwise_cnn" and legacy_metrics_path.is_file():
        return legacy_metrics_path
    return None


def print_saved_test_metrics(model_path: Path) -> None:
    metrics_path = metrics_path_for(model_path)
    if metrics_path is None:
        print("Saved testing metrics are not available for this model.")
        return

    try:
        report = json.loads(metrics_path.read_text(encoding="utf-8"))
        test_metrics = report["cnn"]["test"]
    except (json.JSONDecodeError, KeyError, TypeError):
        print("Saved testing metrics could not be read for this model.")
        return

    print("\nSaved testing metrics:")
    for label, key in (
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1 score", "f1"),
        ("ROC-AUC", "roc_auc"),
    ):
        value = test_metrics.get(key)
        if isinstance(value, (int, float)):
            print(f"{label}: {value:.3f}")


def choose_test_image() -> Path | None:
    images = sorted(TESTING_DIRECTORY.glob("*.png"))
    if not images:
        print("No PNG images exist in testing/.")
        return None

    print("\nTest images:")
    for number in range(1, len(images) + 1):
        print(f"{number}. Image {number}")

    selected = input("Choose image number: ").strip()
    if not selected.isdigit() or not 1 <= int(selected) <= len(images):
        print("Invalid image number.")
        return None
    return images[int(selected) - 1]


def run_test_sequence() -> None:
    model_path = choose_model()
    if model_path is None:
        return

    print_saved_test_metrics(model_path)
    image_path = choose_test_image()
    if image_path is None:
        return

    expected_label = symbol_from_filename(image_path.name)

    command = [
        sys.executable,
        str(PROJECT_DIRECTORY / "inference.py"),
        str(image_path),
        "--checkpoint",
        str(model_path),
    ]
    if expected_label is not None:
        command.extend(["--expected-label", expected_label])
    subprocess.run(command, cwd=PROJECT_DIRECTORY, check=False)


def run_training_sequence() -> None:
    model_name = input("New model name: ").strip()
    if not MODEL_NAME_PATTERN.fullmatch(model_name):
        print("Use only letters, numbers, underscores, and hyphens for the model name.")
        return

    model_path = ARTIFACTS_DIRECTORY / f"{model_name}.pt"
    if model_path.exists():
        print("That model name already exists. Run the program again and choose another name.")
        return

    command = [
        sys.executable,
        str(PROJECT_DIRECTORY / "train.py"),
        "--training-dir",
        str(PROJECT_DIRECTORY / "training"),
        "--validation-dir",
        str(PROJECT_DIRECTORY / "validation"),
        "--testing-dir",
        str(PROJECT_DIRECTORY / "testing"),
        "--output-dir",
        str(ARTIFACTS_DIRECTORY),
        "--model-name",
        model_name,
    ]
    result = subprocess.run(command, cwd=PROJECT_DIRECTORY, check=False)
    if result.returncode == 0:
        print(f"Training completed. Model saved to artifacts/{model_name}.pt")
        print("Run program.py again to choose this model for testing.")
    else:
        print("Training failed. Read the terminal output, then run program.py again.")


def main() -> None:
    print("Clockwise order classifier")
    print("1. Test image")
    print("2. Train new model")
    choice = input("Choose action: ").strip()

    if choice == "1":
        run_test_sequence()
    elif choice == "2":
        run_training_sequence()
    else:
        print("Invalid action.")


if __name__ == "__main__":
    main()
