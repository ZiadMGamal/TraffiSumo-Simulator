from typing import Any, Callable, Dict, Type


class AlgorithmRegistry:
    _registry: Dict[str, Type] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        def decorator(agent_cls: Type) -> Type:
            cls._registry[name.lower()] = agent_cls
            return agent_cls

        return decorator

    @classmethod
    def get(cls, name: str, **kwargs: Any) -> Any:
        key = name.lower()
        if key not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys()))
            raise KeyError(f"Algorithm '{name}' not found. Available: {available}")
        return cls._registry[key](**kwargs)

    @classmethod
    def list_algorithms(cls) -> list:
        return sorted(cls._registry.keys())


class EnvironmentRegistry:
    _registry: Dict[str, Type] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        def decorator(env_cls: Type) -> Type:
            cls._registry[name.lower()] = env_cls
            return env_cls

        return decorator

    @classmethod
    def get(cls, name: str, **kwargs: Any) -> Any:
        key = name.lower()
        if key not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys()))
            raise KeyError(f"Environment '{name}' not found. Available: {available}")
        return cls._registry[key](**kwargs)

    @classmethod
    def list_environments(cls) -> list:
        return sorted(cls._registry.keys())
