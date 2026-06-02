from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch


class BaseMARLAgent(ABC):
    def __init__(self, agent_id: str, state_dim: int, action_dim: int):
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.training_step = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def choose_action(
        self,
        state: np.ndarray,
        explore: bool = True,
        global_state: Optional[np.ndarray] = None,
    ) -> Union[int, np.ndarray]:
        pass

    @abstractmethod
    def update(self, batch: Any) -> Optional[float]:
        pass

    def store_transition(
        self,
        state: np.ndarray,
        action: Union[int, np.ndarray],
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        pass

    def sync_target(self) -> None:
        pass

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass

    def set_eval_mode(self) -> None:
        pass

    def set_train_mode(self) -> None:
        pass


class MultiAgentCoordinator:
    def __init__(self, agents: Dict[str, BaseMARLAgent]):
        self.agents = agents
        self.agent_ids = list(agents.keys())

    def choose_actions(
        self,
        observations: Dict[str, np.ndarray],
        explore: bool = True,
        global_state: Optional[np.ndarray] = None,
    ) -> Dict[str, Union[int, np.ndarray]]:
        return {
            aid: self.agents[aid].choose_action(
                observations[aid], explore=explore, global_state=global_state
            )
            for aid in self.agent_ids
        }

    def update_all(self, batch_size: int) -> Dict[str, Optional[float]]:
        losses = {}
        for aid in self.agent_ids:
            loss = self.agents[aid].update(batch_size)
            losses[aid] = loss
        return losses

    def sync_targets(self) -> None:
        for agent in self.agents.values():
            agent.sync_target()

    def save_all(self, directory: str) -> None:
        for aid, agent in self.agents.items():
            agent.save(f"{directory}/{aid}")

    def load_all(self, directory: str) -> None:
        for aid, agent in self.agents.items():
            agent.load(f"{directory}/{aid}")

    def set_eval_mode(self) -> None:
        for agent in self.agents.values():
            agent.set_eval_mode()

    def set_train_mode(self) -> None:
        for agent in self.agents.values():
            agent.set_train_mode()
