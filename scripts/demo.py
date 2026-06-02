import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bootstrap
from core.env_loader import load_environment
from agents.factory import create_agents
from agents.base_agent import MultiAgentCoordinator


def main():
    env = load_environment("mock", max_steps=200)
    agents = create_agents(env.agent_ids, algorithm="dqn")
    coordinator = MultiAgentCoordinator(agents)
    obs, info = env.reset(seed=42)
    print(f"Demo mode: {info.get('mode', 'mock')}")
    print(f"Intersections: {env.agent_ids}")
    total_steps = 0
    while True:
        actions = coordinator.choose_actions(obs, explore=True)
        obs, rewards, dones, _, metrics = env.step(actions)
        total_steps += 1
        if total_steps % 20 == 0:
            avg_reward = sum(rewards.values()) / len(rewards)
            avg_queue = sum(
                metrics[aid]["queue_length"] for aid in env.agent_ids if aid in metrics
            ) / len(env.agent_ids)
            print(f"Step {total_steps} | Reward {avg_reward:.2f} | Queue {avg_queue:.1f}")
        if any(dones.values()):
            break
    print(f"Episode finished in {total_steps} steps")
    env.close()


if __name__ == "__main__":
    main()
