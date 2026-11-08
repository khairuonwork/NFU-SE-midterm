"""Shared class definitions for the clockwise-order classifier.

These mappings define the supervised targets used while training. They do not
contain an image-reading algorithm. The CNN must learn to associate image pixels
with these targets from the examples in the training folder.
"""

from __future__ import annotations

import re
from typing import Final


POSITIVE_CLASS_ID: Final = 1
NEGATIVE_CLASS_ID: Final = 0
POSITIVE_SYMBOL: Final = "+"
NEGATIVE_SYMBOL: Final = "-"

# This is the class convention used throughout the project:
# A -> B -> C clockwise means positive (+), while A -> C -> B means negative (-).
CLOCKWISE_ORDER_TO_CLASS_ID: Final = {
    "ABC": POSITIVE_CLASS_ID,
    "ACB": NEGATIVE_CLASS_ID,
}

SYMBOL_TO_CLASS_ID: Final = {
    POSITIVE_SYMBOL: POSITIVE_CLASS_ID,
    NEGATIVE_SYMBOL: NEGATIVE_CLASS_ID,
}

# Differentiation
SYMBOL_TO_CLOCKWISE_ORDER: Final = {
    POSITIVE_SYMBOL: "ABC",
    NEGATIVE_SYMBOL: "ACB",
}

LABELLED_IMAGE_FILENAME = re.compile(r"^\d+_(?P<symbol>[+-])\.png$")

# 1 -> +, 0 -> -
def class_id_from_symbol(symbol: str) -> int:
    """Return the binary training target for a `+` or `-` filename suffix."""
    try:
        return SYMBOL_TO_CLASS_ID[symbol]
    except KeyError as error:
        raise ValueError(f"Unsupported class symbol: {symbol}") from error


def symbol_from_filename(filename: str) -> str | None:
    """Return `+` or `-` when a filename follows the labelled dataset format."""
    match = LABELLED_IMAGE_FILENAME.fullmatch(filename)
    return match.group("symbol") if match else None


def class_description(symbol: str) -> str:
    """Return a human-readable description for a class symbol."""
    try:
        clockwise_order = SYMBOL_TO_CLOCKWISE_ORDER[symbol]
    except KeyError as error:
        raise ValueError(f"Unsupported class symbol: {symbol}") from error
    return f"{symbol} (clockwise {clockwise_order})"
