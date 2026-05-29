"""Persistence and validation of the canonical class-name list.

The class list pairs each model output index with a human-readable label. It is
written once at training time and read at inference time; if it desyncs from the
model's output ordering, predictions are silently wrong. The validation here
fails loudly instead.
"""

import json
from pathlib import Path

NUM_CLASSES: int = 39  # 38 PlantVillage classes + 1 Background_without_leaves (negative) class
CLASS_NAMES_PATH: Path = Path("artifacts/checkpoints/class_names.json")


def save_class_names(classes: list[str]) -> None:
    """Persist the alphabetical class list to the canonical path.

    Args:
        classes: The ``ImageFolder``-ordered (alphabetical) class names.

    Raises:
        ValueError: If ``classes`` does not contain exactly ``NUM_CLASSES``
            entries.
    """
    if len(classes) != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} classes, got {len(classes)}: {classes!r}"
        )
    CLASS_NAMES_PATH.parent.mkdir(parents=True, exist_ok=True)
    CLASS_NAMES_PATH.write_text(json.dumps(classes, indent=2))


def load_class_names() -> list[str]:
    """Load the class list, failing loudly if missing or malformed.

    Returns:
        The list of ``NUM_CLASSES`` class names in their persisted order.

    Raises:
        FileNotFoundError: If the canonical file does not exist.
        RuntimeError: If the file is not a list of exactly ``NUM_CLASSES``
            strings.
    """
    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"{CLASS_NAMES_PATH} missing — did you run training first?"
        )
    classes = json.loads(CLASS_NAMES_PATH.read_text())
    if not isinstance(classes, list) or len(classes) != NUM_CLASSES:
        raise RuntimeError(
            f"{CLASS_NAMES_PATH} is malformed: expected list of "
            f"{NUM_CLASSES} strings, got {classes!r}"
        )
    return classes
