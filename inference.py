"""Run one saved CNN checkpoint on one figure image."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import torch
from PIL import Image
from torch import Tensor
from torchvision import transforms

from labels import NEGATIVE_SYMBOL, POSITIVE_SYMBOL, class_description
from model import ClockwiseCNN

def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_model(checkpoint_path: Path, device: torch.device) -> tuple[ClockwiseCNN, dict]:
    """Load one checkpoint and return its CNN in evaluation mode."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = ClockwiseCNN().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint

def preprocessing_from_checkpoint(checkpoint: dict) -> transforms.Compose:
    """Recreate the evaluation preprocessing saved with a trained model."""
    return transforms.Compose(
        [
            transforms.Resize((checkpoint["image_size"], checkpoint["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize((checkpoint["mean"],), (checkpoint["std"],)),
        ]
    )

def positive_probability_for_image(
    image_path: Path,
    model: ClockwiseCNN,
    checkpoint: dict,
    device: torch.device,
) -> float:
    """Return the model probability that one image belongs to the `+` class."""
    preprocess = preprocessing_from_checkpoint(checkpoint)
    with Image.open(image_path) as image:
        image_tensor = cast(Tensor, preprocess(image.convert("L")))
        image_batch = image_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        logit = model(image_batch).squeeze()
    return float(torch.sigmoid(logit).item())

def predict(image_path: Path, checkpoint_path: Path, expected_label: str | None = None) -> None:
    """Print a prediction and, when provided, compare it with a test label."""
    device = choose_device()
    model, checkpoint = load_model(checkpoint_path, device)
    positive_probability = positive_probability_for_image(image_path, model, checkpoint, device)

    if positive_probability >= 0.5:
        predicted_label = POSITIVE_SYMBOL
        confidence = positive_probability
    else:
        predicted_label = NEGATIVE_SYMBOL
        confidence = 1 - positive_probability

    print(f"Prediction: {class_description(predicted_label)}")
    print(f"Confidence for predicted class: {confidence:.3f}")
    print(f"Probability for +: {positive_probability:.3f}")
    if expected_label is not None:
        result = "correct" if predicted_label == expected_label else "incorrect"
        print(f"Expected: {class_description(expected_label)}")
        print(f"Test result: {result}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/clockwise_cnn.pt"))
    parser.add_argument("--expected-label", choices=("+", "-"))
    args = parser.parse_args()
    predict(args.image, args.checkpoint, args.expected_label)
