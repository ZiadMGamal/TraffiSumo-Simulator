import json
from pathlib import Path
from typing import Dict, List, Optional

from core.config import get_settings


class ModelService:
    def __init__(self, model_dir: Optional[str] = None):
        settings = get_settings()
        self.model_dir = Path(model_dir or settings.model_dir)
        self.checkpoint_dir = Path(settings.checkpoint_dir)

    def list_models(self) -> List[Dict]:
        results = []
        if not self.model_dir.exists():
            return results
        for path in sorted(self.model_dir.iterdir()):
            if path.is_file() and path.suffix in (".pt", ".pth"):
                results.append(
                    {
                        "name": path.name,
                        "path": str(path),
                        "size_bytes": path.stat().st_size,
                        "type": "policy",
                    }
                )
            elif path.is_dir():
                files = list(path.glob("*"))
                results.append(
                    {
                        "name": path.name,
                        "path": str(path),
                        "files": [f.name for f in files],
                        "type": "agent_bundle",
                    }
                )
        return results

    def list_checkpoints(self) -> List[Dict]:
        if not self.checkpoint_dir.exists():
            return []
        checkpoints = []
        for path in sorted(self.checkpoint_dir.glob("ep*")):
            meta_path = path / "metadata.json"
            meta = {}
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            checkpoints.append(
                {
                    "name": path.name,
                    "path": str(path),
                    "episode": meta.get("episode"),
                    "metrics": meta.get("metrics"),
                }
            )
        return checkpoints

    def get_model_info(self) -> Dict:
        models = self.list_models()
        checkpoints = self.list_checkpoints()
        return {
            "model_dir": str(self.model_dir),
            "model_count": len(models),
            "checkpoint_count": len(checkpoints),
            "models": models,
            "checkpoints": checkpoints[:10],
        }
