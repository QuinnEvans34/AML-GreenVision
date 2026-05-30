"""Treatment knowledge base loader and lookup helpers.

See ``IMPLEMENTATION_GUIDE.md`` Decision 9 for the KB design rationale.
The JSON file is the single source of truth; this module just loads and
indexes it at startup.
"""

from __future__ import annotations

import json
from pathlib import Path

from api.schemas import Treatment

TREATMENTS_PATH = Path("data/treatments.json")


def load_treatments() -> tuple[dict[str, Treatment], dict]:
    """Load ``data/treatments.json`` and split metadata from entries.

    Returns:
        Tuple ``(entries, metadata)`` where ``entries`` is a dict mapping
        raw class name → ``Treatment`` and ``metadata`` is the top-level
        ``_metadata`` block (version, disclaimer, etc.).

    Raises:
        FileNotFoundError: If the KB file is missing.
    """
    if not TREATMENTS_PATH.exists():
        raise FileNotFoundError(
            f"Treatment KB not found at {TREATMENTS_PATH}. "
            "Was data/treatments.json checked into the repo?"
        )
    with TREATMENTS_PATH.open() as f:
        raw = json.load(f)
    metadata = raw.pop("_metadata", {})
    entries = {key: Treatment(**val) for key, val in raw.items()}
    return entries, metadata


def get_treatment(treatments: dict[str, Treatment], class_name: str) -> Treatment:
    """Look up the treatment entry for a predicted class.

    Args:
        treatments: The dict returned by ``load_treatments()``.
        class_name: The raw class string from the model output.

    Raises:
        KeyError: If the class has no treatment entry. This indicates a
            mismatch between the registered model and the KB and should
            surface as a 500 to the client (not a 404).

    Returns:
        The ``Treatment`` for the class.
    """
    if class_name not in treatments:
        raise KeyError(
            f"No treatment entry for class: {class_name}. "
            "Treatment KB and model class names are out of sync."
        )
    return treatments[class_name]
