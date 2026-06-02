import torch
import pytest

from agents.networks import DuelingQNetwork, MixingNetwork, PPONetwork


def test_dueling_q_network_forward():
    net = DuelingQNetwork(26, 4)
    state = torch.randn(8, 26)
    out = net(state)
    assert out.shape == (8, 4)


def test_mixing_network():
    mixer = MixingNetwork(num_agents=4, state_dim=52)
    agent_qs = torch.randn(8, 1, 4)
    global_state = torch.randn(8, 52)
    out = mixer(agent_qs, global_state)
    assert out.shape == (8, 1, 1)


def test_ppo_network():
    net = PPONetwork(26, 4)
    state = torch.randn(4, 26)
    logits, value = net(state)
    assert logits.shape == (4, 4)
    assert value.shape == (4, 1)
