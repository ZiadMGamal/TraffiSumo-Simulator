import os
import random
from typing import Any, Optional

import numpy as np
import torch
import torch.optim as optim

from agents.base_agent import BaseMARLAgent
from agents.exploration import EpsilonGreedy
from agents.networks import DuelingQNetwork
from agents.replay_buffer import CircularReplayBuffer
from core.registry import AlgorithmRegistry
from core.utils import hard_update


@AlgorithmRegistry.register("dqn")
class DQNAgent(BaseMARLAgent):
    def __init__(
        self,
        agent_id: str = "agent",
        state_dim: int = 26,
        action_dim: int = 4,
        lr: float = 1e-4,
        gamma: float = 0.99,
        buffer_capacity: int = 100000,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.9995,
    ):
        super().__init__(agent_id, state_dim, action_dim)
        self.gamma = gamma
        self.explorer = EpsilonGreedy(
            action_dim, epsilon_start, epsilon_min, epsilon_decay
        )
        self.policy_net = DuelingQNetwork(state_dim, action_dim).to(self.device)
        self.target_net = DuelingQNetwork(state_dim, action_dim).to(self.device)
        hard_update(self.target_net, self.policy_net)
        self.optimizer = optim.AdamW(
            self.policy_net.parameters(), lr=lr, weight_decay=1e-5
        )
        self.memory = CircularReplayBuffer(buffer_capacity)

    def choose_action(
        self,
        state: np.ndarray,
        explore: bool = True,
        global_state: Optional[np.ndarray] = None,
    ) -> int:
        if explore and random.random() < self.explorer.epsilon:
            return random.randint(0, self.action_dim - 1)
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return int(torch.argmax(self.policy_net(state_t)).item())

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
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        states_t = torch.tensor(states, device=self.device)
        actions_t = torch.tensor(actions, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, device=self.device).unsqueeze(1)
        next_states_t = torch.tensor(next_states, device=self.device)
        dones_t = torch.tensor(dones, device=self.device).unsqueeze(1)
        q_values = self.policy_net(states_t).gather(1, actions_t)
        with torch.no_grad():
            next_actions = self.policy_net(next_states_t).max(1)[1].unsqueeze(1)
            next_q = self.target_net(next_states_t).gather(1, next_actions)
            expected = rewards_t + self.gamma * next_q * (1 - dones_t)
        loss = torch.nn.functional.mse_loss(q_values, expected)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        self.explorer.decay()
        self.training_step += 1
        return float(loss.item())

    def sync_target(self) -> None:
        hard_update(self.target_net, self.policy_net)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "policy": self.policy_net.state_dict(),
                "target": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.explorer.epsilon,
                "training_step": self.training_step,
            },
            f"{path}_dqn.pt",
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(f"{path}_dqn.pt", map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy"])
        self.target_net.load_state_dict(checkpoint["target"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.explorer.epsilon = checkpoint.get("epsilon", 0.0)
        self.training_step = checkpoint.get("training_step", 0)

    def set_eval_mode(self) -> None:
        self.policy_net.eval()
        self.explorer.set_epsilon(0.0)

    def set_train_mode(self) -> None:
        self.policy_net.train()
