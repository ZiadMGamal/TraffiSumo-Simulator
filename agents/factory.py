from typing import Dict, List, Optional

from agents.base_agent import BaseMARLAgent
from core.config import get_settings
from core.registry import AlgorithmRegistry


def create_agent(
    agent_id: str,
    algorithm: Optional[str] = None,
    state_dim: Optional[int] = None,
    action_dim: Optional[int] = None,
    **kwargs,
) -> BaseMARLAgent:
    settings = get_settings()
    return AlgorithmRegistry.get(
        algorithm or settings.algorithm,
        agent_id=agent_id,
        state_dim=state_dim or settings.state_dim,
        action_dim=action_dim or settings.action_dim,
        buffer_capacity=settings.buffer_capacity,
        gamma=settings.gamma,
        lr=settings.learning_rate,
        **kwargs,
    )


def create_agents(
    agent_ids: List[str],
    algorithm: Optional[str] = None,
    **kwargs,
) -> Dict[str, BaseMARLAgent]:
    return {aid: create_agent(aid, algorithm=algorithm, **kwargs) for aid in agent_ids}
