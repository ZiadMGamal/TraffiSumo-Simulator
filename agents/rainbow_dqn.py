import os
from typing import Optional

import numpy as np
import torch
import torch.optim as optim

from agents.base_agent import BaseMARLAgent
from agents.networks import RainbowNetwork
from agents.replay_buffer import PrioritizedReplayBuffer
from core.registry import AlgorithmRegistry
from core.utils import hard_update


@AlgorithmRegistry.register("rainbow")
class RainbowDQNAgent(BaseMARLAgent):
    def __init__(
        self,
        agent_id: str = "agent",
        state_dim: int = 26,
        action_dim: int = 4,
        lr: float = 6.25e-5,
        gamma: float = 0.99,
        buffer_capacity: int = 100000,
        num_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
    ):
        super().__init__(agent_id, state_dim, action_dim)
        self.gamma = gamma
        self.num_atoms = num_atoms
        self.policy_net = RainbowNetwork(
            state_dim, action_dim, num_atoms, v_min, v_max
        ).to(self.device)
        self.target_net = RainbowNetwork(
            state_dim, action_dim, num_atoms, v_min, v_max
        ).to(self.device)
        hard_update(self.target_net, self.policy_net)
        self.support = self.policy_net.support
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = PrioritizedReplayBuffer(buffer_capacity)
        self.delta_z = (v_max - v_min) / (num_atoms - 1)

    def _project_distribution(
        self, rewards: torch.Tensor, next_dist: torch.Tensor, dones: torch.Tensor
    ) -> torch.Tensor:
        batch = rewards.size(0)
        delta_z = self.delta_z
        support = self.support
        tz = rewards + (1 - dones) * self.gamma * support
        tz = tz.clamp(self.support[0], self.support[-1])
        b = (tz - support[0]) / delta_z
        lower = b.floor().long()
        upper = b.ceil().long()
        proj = torch.zeros(batch, self.num_atoms, device=self.device)
        offset = (
            torch.linspace(0, (batch - 1) * self.num_atoms, batch, device=self.device)
            .long()
            .unsqueeze(1)
            .expand(batch, self.num_atoms)
        )
        proj.view(-1).index_add_(
            0,
            (lower + offset).view(-1),
            (next_dist * (upper.float() - b)).view(-1),
        )
        proj.view(-1).index_add_(
            0,
            (upper + offset).view(-1),
            (next_dist * (b - lower.float())).view(-1),
        )
        return proj

    def choose_action(
        self,
        state: np.ndarray,
        explore: bool = True,
        global_state: Optional[np.ndarray] = None,
    ) -> int:
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            dist = self.policy_net(state_t)
            q = (dist * self.support).sum(dim=2)
            return int(q.argmax(dim=1).item())

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.memory.push(state, action, reward, next_state, done)

    def update(self, batch_size: int) -> Optional[float]:
        if len(self.memory) < batch_size:
            return None
        states, actions, rewards, next_states, dones, indices, weights = (
            self.memory.sample(batch_size)
        )
        states_t = torch.tensor(states, device=self.device)
        actions_t = torch.tensor(actions, device=self.device)
        rewards_t = torch.tensor(rewards, device=self.device)
        next_states_t = torch.tensor(next_states, device=self.device)
        dones_t = torch.tensor(dones, device=self.device)
        weights_t = torch.tensor(weights, device=self.device)
        batch = states_t.size(0)
        dist = self.policy_net(states_t)
        action_dist = dist[range(batch), actions_t]
        with torch.no_grad():
            next_dist = self.target_net(next_states_t)
            next_actions = (next_dist * self.support).sum(2).argmax(1)
            next_dist = next_dist[range(batch), next_actions]
            proj = self._project_distribution(rewards_t, next_dist, dones_t)
        log_prob = (action_dist * proj).sum(1).log()
        loss = -(log_prob * weights_t).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.policy_net.reset_noise()
        self.training_step += 1
        priorities = loss.detach().abs().cpu().numpy() + 1e-6
        self.memory.update_priorities(indices, priorities)
        return float(loss.item())

    def sync_target(self) -> None:
        hard_update(self.target_net, self.policy_net)

    def save(self, path: str) -> None:
        torch.save(self.policy_net.state_dict(), f"{path}_rainbow.pt")

    def load(self, path: str) -> None:
        self.policy_net.load_state_dict(
            torch.load(f"{path}_rainbow.pt", map_location=self.device)
        )
