import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from core.utils import ensure_dir


class CheckpointManager:
    def __init__(self, checkpoint_dir: str = "checkpoints", max_keep: int = 5):
        self.checkpoint_dir = ensure_dir(checkpoint_dir)
        self.max_keep = max_keep
        self.checkpoints: list = []

    def save(
        self,
        agents_save_fn,
        episode: int,
        metrics: Dict[str, Any],
        tag: Optional[str] = None,
    ) -> Path:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        name = f"ep{episode:05d}_{timestamp}"
        if tag:
            name = f"{name}_{tag}"
        path = self.checkpoint_dir / name
        path.mkdir(parents=True, exist_ok=True)
        agents_save_fn(str(path))
        meta = {"episode": episode, "metrics": metrics, "path": str(path)}
        with open(path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        self.checkpoints.append(path)
        self._prune_old()
        return path

    def _prune_old(self) -> None:
        while len(self.checkpoints) > self.max_keep:
            old = self.checkpoints.pop(0)
            if old.exists():
                shutil.rmtree(old, ignore_errors=True)

    def load_latest(self, agents_load_fn) -> Optional[Dict[str, Any]]:
        checkpoints = sorted(self.checkpoint_dir.glob("ep*"))
        if not checkpoints:
            return None
        latest = checkpoints[-1]
        agents_load_fn(str(latest))
        meta_path = latest / "metadata.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"path": str(latest)}

    def list_checkpoints(self) -> list:
        return sorted([p.name for p in self.checkpoint_dir.glob("ep*")])
