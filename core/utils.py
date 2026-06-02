import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: Any, path: Union[str, Path]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: Union[str, Path]) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def moving_average(values: List[float], window: int = 10) -> List[float]:
    if not values:
        return []
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(sum(values[start : i + 1]) / (i - start + 1))
    return result


def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            tau * source_param.data + (1.0 - tau) * target_param.data
        )


def hard_update(target: torch.nn.Module, source: torch.nn.Module) -> None:
    target.load_state_dict(source.state_dict())


def to_tensor(
    data: np.ndarray,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return torch.tensor(data, dtype=dtype, device=device)


def aggregate_metrics(
    metrics_list: List[Dict[str, float]],
    keys: Optional[List[str]] = None,
) -> Dict[str, float]:
    if not metrics_list:
        return {}
    keys = keys or list(metrics_list[0].keys())
    return {k: float(np.mean([m[k] for m in metrics_list if k in m])) for k in keys}
