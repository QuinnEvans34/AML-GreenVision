"""Promote a trained model version to Production.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/promote.py --version 3
"""

import argparse

from greenvision.training.mlflow_utils import init_mlflow
from greenvision.training.registry import (
    list_versions,
    promote_to_production,
    verify_production_loads,
)


def main() -> None:
    """Promote the requested GreenVision version to Production and verify it."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version",
        type=int,
        required=True,
        help="GreenVision model version to promote",
    )
    args = parser.parse_args()

    init_mlflow()
    print("Current versions:", [(v.version, v.current_stage) for v in list_versions()])
    promote_to_production(args.version)
    verify_production_loads()
    print(f"✓ GreenVision v{args.version} → Production")


if __name__ == "__main__":
    main()
