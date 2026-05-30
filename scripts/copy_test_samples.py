"""Copy one held-out test image per class to web/public/test-samples/.

Reproduces the exact same seed=42 stratified 80/10/10 split as training so
the images copied here are guaranteed to be in the model's held-out test
partition — the model has never seen them.

Output:
    web/public/test-samples/<class_name>/<filename>     (one image per class)
    web/public/test-samples.json                        (manifest the UI reads)

Usage:
    PYTHONPATH=src .venv/bin/python scripts/copy_test_samples.py
    PYTHONPATH=src .venv/bin/python scripts/copy_test_samples.py --per-class 2
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

from greenvision.data.datasets import build_dataloaders


DEFAULT_DATA_ROOT = "data/raw/PlantVillage"
DEFAULT_OUTPUT_DIR = Path("web/public/test-samples")
DEFAULT_MANIFEST = Path("web/public/test-samples.json")


def format_display(raw: str) -> str:
    """Mirror of web/lib/format-class-name.ts."""
    if raw == "Background_without_leaves":
        return "No leaf detected"
    crop_raw, _, condition_raw = raw.partition("___")
    crop = crop_raw.replace("_", " ").strip()
    if crop == "Pepper, bell":
        crop = "Bell pepper"
    if condition_raw == "healthy":
        return f"{crop} (healthy)"
    condition = condition_raw.replace("_", " ").strip()
    return f"{crop} — {condition}"


def crop_of(raw: str) -> str:
    """Top-level grouping for the UI tabs."""
    if raw == "Background_without_leaves":
        return "Background"
    crop = raw.partition("___")[0].replace("_", " ").strip()
    if crop == "Pepper, bell":
        return "Bell pepper"
    return crop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default=DEFAULT_DATA_ROOT,
        help=f"PlantVillage root (default: {DEFAULT_DATA_ROOT})",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Where to copy images (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help=f"Manifest JSON path (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=1,
        help="How many test images to copy per class (default: 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used during training — must match for the split to be reproducible",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest)

    print(f"→ Loading test split from {args.data_root} (seed={args.seed})…")
    _, _, test_loader, class_names = build_dataloaders(
        args.data_root, seed=args.seed, num_workers=0
    )
    test_subset = test_loader.dataset
    imgfolder = test_subset.dataset  # the underlying ImageFolder
    print(f"  {len(class_names)} classes · {len(test_subset)} test images")

    # Group test indices by class
    indices_by_class: dict[int, list[str]] = defaultdict(list)
    for idx in test_subset.indices:
        path, class_idx = imgfolder.samples[idx]
        indices_by_class[class_idx].append(path)

    # ── Wipe and rebuild the output dir ─────────────────────────────
    if output_dir.exists():
        print(f"→ Clearing existing {output_dir}…")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Copy N samples per class, build manifest ───────────────────
    manifest: list[dict] = []
    total_copied = 0
    for class_idx in sorted(indices_by_class.keys()):
        class_name = class_names[class_idx]
        candidates = indices_by_class[class_idx]
        # Random.sample is reproducible because of the seed set above
        chosen = random.sample(candidates, min(args.per_class, len(candidates)))

        class_out = output_dir / class_name
        class_out.mkdir(parents=True, exist_ok=True)

        urls: list[str] = []
        for src_path in chosen:
            src = Path(src_path)
            dest = class_out / src.name
            shutil.copy2(src, dest)
            # URL relative to web/public/
            rel = dest.relative_to(Path("web/public"))
            urls.append(f"/{rel.as_posix()}")
            total_copied += 1

        entry = {
            "class_name": class_name,
            "display_name": format_display(class_name),
            "crop": crop_of(class_name),
            "is_healthy": class_name.endswith("___healthy"),
            "is_background": class_name == "Background_without_leaves",
            "images": urls,
        }
        manifest.append(entry)

    # Sort manifest: background first, then alphabetical by crop, healthy last within crop
    def sort_key(e: dict):
        if e["is_background"]:
            return (0, "", 0)
        return (1, e["crop"], 1 if e["is_healthy"] else 0)

    manifest.sort(key=sort_key)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)

    print()
    print(f"✓ Copied {total_copied} test images to {output_dir}")
    print(f"✓ Wrote manifest → {manifest_path}")
    print(f"  ({len(manifest)} class entries · {args.per_class} image(s) each)")


if __name__ == "__main__":
    main()
