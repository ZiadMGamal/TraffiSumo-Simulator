from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingQNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.feature_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.feature_network(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        return values + (advantages - advantages.mean(dim=-1, keepdim=True))


class NoisyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))
        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        mu_range = 1 / self.in_features**0.5
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / self.in_features**0.5)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / self.in_features**0.5)

    def reset_noise(self) -> None:
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size: int) -> torch.Tensor:
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign() * x.abs().sqrt()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class RainbowNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.num_atoms = num_atoms
        self.register_buffer("support", torch.linspace(v_min, v_max, num_atoms))

        self.feature = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.value = nn.Sequential(
            NoisyLinear(256, 128),
            nn.ReLU(),
            NoisyLinear(128, num_atoms),
        )
        self.advantage = nn.Sequential(
            NoisyLinear(256, 128),
            nn.ReLU(),
            NoisyLinear(128, output_dim * num_atoms),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        batch = state.size(0)
        features = self.feature(state)
        value = self.value(features).view(batch, 1, self.num_atoms)
        advantage = self.advantage(features).view(
            batch, self.output_dim, self.num_atoms
        )
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        return F.softmax(q_atoms, dim=-1)

    def reset_noise(self) -> None:
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


class ActorNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class CriticNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_agents: int = 1,
        hidden_dim: int = 256,
    ):
        super().__init__()
        input_dim = state_dim * num_agents + action_dim * num_agents
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([states, actions], dim=-1)
        return self.net(x)


class MixingNetwork(nn.Module):
    def __init__(self, num_agents: int, state_dim: int, embed_dim: int = 32):
        super().__init__()
        self.num_agents = num_agents
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_agents * embed_dim),
        )
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(
        self, agent_qs: torch.Tensor, global_state: torch.Tensor
    ) -> torch.Tensor:
        batch = agent_qs.size(0)
        w1 = torch.abs(self.hyper_w1(global_state)).view(batch, self.num_agents, -1)
        b1 = self.hyper_b1(global_state).view(batch, 1, -1)
        hidden = F.elu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)
        w2 = torch.abs(self.hyper_w2(global_state)).view(batch, -1, 1)
        b2 = self.hyper_b2(global_state).view(batch, 1, 1)
        return torch.bmm(hidden, w2) + b2


class RNNAgentNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(
        self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out, hidden = self.gru(x, hidden)
        q = self.fc(out[:, -1, :])
        return q, hidden


class AttentionEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.projection = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.output = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.projection(x)
        attended, _ = self.attention(projected, projected, projected)
        return self.output(attended.mean(dim=1))


class PPONetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.policy = nn.Linear(hidden_dim, action_dim)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(state)
        return self.policy(features), self.value(features)
