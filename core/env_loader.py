import os
from typing import Optional, Union

from core.config import get_settings
from core.registry import EnvironmentRegistry


def load_environment(
    name: Optional[str] = None,
    config_file: Optional[str] = None,
    use_gui: bool = False,
    fallback_mock: bool = True,
):
    settings = get_settings()
    env_name = name or os.environ.get("ENV_MODE", "auto")
    if env_name == "auto":
        env_name = "sumo" if os.environ.get("SUMO_HOME") else "mock"
    try:
        if env_name == "sumo":
            return EnvironmentRegistry.get(
                "sumo",
                config_file=config_file,
                use_gui=use_gui,
            )
        return EnvironmentRegistry.get(env_name)
    except (RuntimeError, KeyError) as e:
        if fallback_mock and env_name == "sumo":
            return EnvironmentRegistry.get("mock")
        raise e
