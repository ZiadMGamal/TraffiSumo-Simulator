import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.optim as optim

from agents.base_agent import BaseMARLAgent
from agents.exploration import EpsilonGreedy
from agents.networks import DuelingQNetwork, MixingNetwork, RNNAgentNetwork
from agents.replay_buffer import MultiAgentReplayBuffer
from core.registry import AlgorithmRegistry
from core.utils import hard_update


@AlgorithmRegistry.register("qmix")
class QMIXAgent(BaseMARLAgent):
    def __init__(
        self,
        agent_id: str = "agent",
        state_dim: int = 26,
        action_dim: int = 4,
        global_state_dim: int = 52,
        num_agents: int = 4,
        lr: float = 5e-4,
        gamma: float = 0.99,
        buffer_capacity: int = 50000,
        use_rnn: bool = False,
    ):
        super().__init__(agent_id, state_dim, action_dim)
        self.global_state_dim = global_state_dim
        self.num_agents = num_agents
        self.gamma = gamma
        self.use_rnn = use_rnn
        if use_rnn:
            self.q_net = RNNAgentNetwork(state_dim, action_dim).to(self.device)
            self.target_q = RNNAgentNetwork(state_dim, action_dim).to(self.device)
        else:
            self.q_net = DuelingQNetwork(state_dim, action_dim).to(self.device)
            self.target_q = DuelingQNetwork(state_dim, action_dim).to(self.device)
        hard_update(self.target_q, self.q_net)
        self.mixer = MixingNetwork(num_agents, global_state_dim).to(self.device)
        self.target_mixer = MixingNetwork(num_agents, global_state_dim).to(
            self.device
        )
        hard_update(self.target_mixer, self.mixer)
        params = list(self.q_net.parameters()) + list(self.mixer.parameters())
        self.optimizer = optim.RMSprop(params, lr=lr, alpha=0.99, eps=1e-5)
        self.explorer = EpsilonGreedy(action_dim)
        self.hidden = None

    def choose_action(
        self,
        state: np.ndarray,
        explore: bool = True,
        global_state: Optional[np.ndarray] = None,
    ) -> int:
        if explore and np.random.random() < self.explorer.epsilon:
            return int(np.random.randint(0, self.action_dim))
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if self.use_rnn:
                state_seq = state_t.unsqueeze(1)
                q, self.hidden = self.q_net(state_seq, self.hidden)
            else:
                q = self.q_net(state_t)
            return int(q.argmax(dim=-1).item())

    def compute_q_values(self, states: torch.Tensor) -> torch.Tensor:
        if self.use_rnn:
            q, _ = self.q_net(states.unsqueeze(1))
        else:
            q = self.q_net(states)
        return q

    def mix_values(
        self, agent_qs: torch.Tensor, global_state: torch.Tensor
    ) -> torch.Tensor:
        return self.mixer(agent_qs, global_state)

    def update(self, batch: dict) -> Optional[float]:
        if not batch:
            return None
        states = torch.tensor(batch["states"], device=self.device)
        actions = torch.tensor(batch["actions"], device=self.device).long()
        rewards = torch.tensor(batch["rewards"], device=self.device)
        next_states = torch.tensor(batch["next_states"], device=self.device)
        global_states = torch.tensor(batch["global_states"], device=self.device)
        global_next = torch.tensor(batch["global_next_states"], device=self.device)
        dones = torch.tensor(batch["dones"], device=self.device)
        q_values = self.compute_q_values(states)
        chosen_q = q_values.gather(1, actions.unsqueeze(1))
        with torch.no_grad():
            next_q = self.target_q(next_states).max(dim=1)[0]
            agent_qs = chosen_q
            target_q_tot = self.target_mixer(
                agent_qs.unsqueeze(1), global_next
            ).squeeze()
            expected = rewards + self.gamma * target_q_tot * (1 - dones)
        q_tot = self.mixer(chosen_q.unsqueeze(1), global_states).squeeze()
        loss = torch.nn.functional.mse_loss(q_tot, expected)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()
        self.explorer.decay()
        self.training_step += 1
        return float(loss.item())

    def sync_target(self) -> None:
        hard_update(self.target_q, self.q_net)
        hard_update(self.target_mixer, self.mixer)

    def save(self, path: str) -> None:
        torch.save(
            {
                "q_net": self.q_net.state_dict(),
                "mixer": self.mixer.state_dict(),
            },
            f"{path}_qmix.pt",
        )

    def load(self, path: str) -> None:
        data = torch.load(f"{path}_qmix.pt", map_location=self.device)
        self.q_net.load_state_dict(data["q_net"])
        self.mixer.load_state_dict(data["mixer"])
