"""Run a saved CNN checkpoint on one new figure image."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import torch
from PIL import Image
from torch import Tensor
from torchvision import transforms

from model import ClockwiseCNN


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def predict(image_path: Path, checkpoint_path: Path) -> None:
    device = choose_device()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = ClockwiseCNN().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    preprocess = transforms.Compose(
        [
            transforms.Resize((checkpoint["image_size"], checkpoint["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize((checkpoint["mean"],), (checkpoint["std"],)),
        ]
    )
    with Image.open(image_path) as image:
        image_tensor = cast(Tensor, preprocess(image.convert("L")))
        batch = image_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        positive_probability = float(torch.sigmoid(model(batch).squeeze()).item())
    if positive_probability >= 0.5:
        prediction = "+ (clockwise ABC)"
        confidence = positive_probability
    else:
        prediction = "- (clockwise ACB)"
        confidence = 1 - positive_probability
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.3f}")
    print(f"Probability for +: {positive_probability:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/clockwise_cnn.pt"))
    args = parser.parse_args()
    predict(args.image, args.checkpoint)
