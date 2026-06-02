import os
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from agents.base_agent import BaseMARLAgent
from agents.networks import PPONetwork
from core.registry import AlgorithmRegistry


@AlgorithmRegistry.register("ppo")
class PPOAgent(BaseMARLAgent):
    def __init__(
        self,
        agent_id: str = "agent",
        state_dim: int = 26,
        action_dim: int = 4,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        ppo_epochs: int = 4,
        mini_batch_size: int = 64,
    ):
        super().__init__(agent_id, state_dim, action_dim)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.network = PPONetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.rollout_states: List[np.ndarray] = []
        self.rollout_actions: List[int] = []
        self.rollout_rewards: List[float] = []
        self.rollout_values: List[float] = []
        self.rollout_log_probs: List[float] = []
        self.rollout_dones: List[bool] = []

    def choose_action(
        self,
        state: np.ndarray,
        explore: bool = True,
        global_state: Optional[np.ndarray] = None,
    ) -> int:
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        logits, value = self.network(state_t)
        probs = F.softmax(logits, dim=-1)
        if explore:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        else:
            action = probs.argmax(dim=-1)
            log_prob = torch.log(probs.gather(1, action.unsqueeze(1))).squeeze()
        self.rollout_states.append(state)
        self.rollout_actions.append(int(action.item()))
        self.rollout_values.append(float(value.item()))
        self.rollout_log_probs.append(float(log_prob.item()))
        return int(action.item())

    def store_reward(self, reward: float, done: bool) -> None:
        self.rollout_rewards.append(reward)
        self.rollout_dones.append(done)

    def _compute_gae(self, next_value: float) -> Tuple[np.ndarray, np.ndarray]:
        rewards = np.array(self.rollout_rewards)
        values = np.array(self.rollout_values + [next_value])
        dones = np.array(self.rollout_dones + [False])
        advantages = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = (
                rewards[t]
                + self.gamma * values[t + 1] * (1 - dones[t])
                - values[t]
            )
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values[:-1]
        return advantages, returns

    def update(self, batch_size: int = 0) -> Optional[float]:
        if len(self.rollout_states) < 2:
            return None
        advantages, returns = self._compute_gae(0.0)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        states = torch.tensor(np.array(self.rollout_states), device=self.device)
        actions = torch.tensor(self.rollout_actions, device=self.device)
        old_log_probs = torch.tensor(
            self.rollout_log_probs, device=self.device
        )
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        adv_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        total_loss = 0.0
        for _ in range(self.ppo_epochs):
            logits, values = self.network(states)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(actions)
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * adv_t
            surr2 = (
                torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                * adv_t
            )
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values.squeeze(), returns_t)
            entropy = dist.entropy().mean()
            loss = (
                policy_loss
                + self.value_coef * value_loss
                - self.entropy_coef * entropy
            )
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()
            total_loss += float(loss.item())
        self._clear_rollout()
        self.training_step += 1
        return total_loss / self.ppo_epochs

    def _clear_rollout(self) -> None:
        self.rollout_states.clear()
        self.rollout_actions.clear()
        self.rollout_rewards.clear()
        self.rollout_values.clear()
        self.rollout_log_probs.clear()
        self.rollout_dones.clear()

    def sync_target(self) -> None:
        pass

    def save(self, path: str) -> None:
        torch.save(self.network.state_dict(), f"{path}_ppo.pt")

    def load(self, path: str) -> None:
        self.network.load_state_dict(
            torch.load(f"{path}_ppo.pt", map_location=self.device)
        )

    def set_eval_mode(self) -> None:
        self.network.eval()

    def set_train_mode(self) -> None:
        self.network.train()
