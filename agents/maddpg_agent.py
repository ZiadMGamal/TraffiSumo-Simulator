import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from agents.base_agent import BaseMARLAgent
from agents.exploration import OrnsteinUhlenbeckNoise
from agents.networks import ActorNetwork, CriticNetwork
from agents.replay_buffer import CircularReplayBuffer
from core.registry import AlgorithmRegistry
from core.utils import soft_update


@AlgorithmRegistry.register("maddpg")
class MADDPGAgent(BaseMARLAgent):
    def __init__(
        self,
        agent_id: str = "agent",
        state_dim: int = 26,
        action_dim: int = 4,
        num_agents: int = 1,
        all_agent_ids: Optional[List[str]] = None,
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.01,
        buffer_capacity: int = 100000,
    ):
        super().__init__(agent_id, state_dim, action_dim)
        self.num_agents = num_agents
        self.all_agent_ids = all_agent_ids or [agent_id]
        self.gamma = gamma
        self.tau = tau
        self.actor = ActorNetwork(state_dim, action_dim).to(self.device)
        self.actor_target = ActorNetwork(state_dim, action_dim).to(self.device)
        self.critic = CriticNetwork(
            state_dim, action_dim, num_agents
        ).to(self.device)
        self.critic_target = CriticNetwork(
            state_dim, action_dim, num_agents
        ).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.memory = CircularReplayBuffer(buffer_capacity)
        self.noise = OrnsteinUhlenbeckNoise(action_dim)
        self.partner_actors: Dict[str, ActorNetwork] = {}

    def register_partners(self, actors: Dict[str, ActorNetwork]) -> None:
        self.partner_actors = actors

    def choose_action(
        self,
        state: np.ndarray,
        explore: bool = True,
        global_state: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(state_t).cpu().numpy()[0]
        if explore:
            action = action + self.noise.sample()
        return np.clip(action, -1.0, 1.0)

    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        discrete_action = int(np.argmax(action)) if action.ndim > 0 else int(action)
        self.memory.push(state, discrete_action, reward, next_state, done)

    def update(self, batch: dict) -> Optional[float]:
        if not batch:
            return None
        states = torch.tensor(batch["states"], device=self.device)
        actions = torch.tensor(batch["actions"], device=self.device)
        rewards = torch.tensor(batch["rewards"], device=self.device).unsqueeze(1)
        next_states = torch.tensor(batch["next_states"], device=self.device)
        dones = torch.tensor(batch["dones"], device=self.device).unsqueeze(1)
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            expected = rewards + self.gamma * target_q * (1 - dones)
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, expected)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)
        self.training_step += 1
        return float(critic_loss.item())

    def sync_target(self) -> None:
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

    def save(self, path: str) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
            },
            f"{path}_maddpg.pt",
        )

    def load(self, path: str) -> None:
        data = torch.load(f"{path}_maddpg.pt", map_location=self.device)
        self.actor.load_state_dict(data["actor"])
        self.critic.load_state_dict(data["critic"])
