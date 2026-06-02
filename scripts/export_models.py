import argparse
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import get_settings
from core.logging import setup_logging
from core.utils import ensure_dir, save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Export trained models for deployment")
    parser.add_argument("--source", type=str, default=None)
    parser.add_argument("--destination", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    settings = get_settings()
    source = Path(args.source or settings.model_dir)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    destination = ensure_dir(args.destination or f"exports/models_{timestamp}")
    logger = setup_logging("export")
    if not source.exists():
        logger.error(f"Source directory not found: {source}")
        return
    for item in source.iterdir():
        if item.is_file():
            shutil.copy2(item, destination / item.name)
        elif item.is_dir():
            shutil.copytree(item, destination / item.name, dirs_exist_ok=True)
    manifest = {
        "exported_at": timestamp,
        "source": str(source),
        "files": [f.name for f in destination.iterdir()],
    }
    save_json(manifest, destination / "manifest.json")
    logger.info(f"Exported models to {destination}")


if __name__ == "__main__":
    main()
