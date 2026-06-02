import asyncio
import json
import os
from typing import Any, Dict, List, Optional

from agents.base_agent import MultiAgentCoordinator
from agents.dqn_agent import DQNAgent
from core.config import get_settings
from core.env_loader import load_environment
from core.logging import get_logger


class SimulationService:
    _instance: Optional["SimulationService"] = None

    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger("simulation.service")
        self.env = None
        self.coordinator: Optional[MultiAgentCoordinator] = None
        self.running = False
        self.use_trained_models = True
        self.mode = "unknown"

    @classmethod
    def get_instance(cls) -> "SimulationService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def initialize(self, use_gui: bool = False, force_mock: bool = False) -> None:
        if self.env is not None:
            self.env.close()
        if force_mock or not os.environ.get("SUMO_HOME"):
            self.env = load_environment("mock")
            self.mode = "mock"
            self.logger.info("Simulation running in mock mode")
        else:
            try:
                self.env = load_environment(
                    "sumo",
                    config_file=str(self.settings.sumo_config_path),
                    use_gui=use_gui,
                    fallback_mock=True,
                )
                self.mode = getattr(self.env, "metadata", {}).get("mode", "sumo")
            except Exception as e:
                self.logger.warning(f"SUMO unavailable, using mock: {e}")
                self.env = load_environment("mock")
                self.mode = "mock"
        agents = {
            aid: DQNAgent(
                agent_id=aid,
                state_dim=self.settings.state_dim,
                action_dim=self.settings.action_dim,
            )
            for aid in self.env.agent_ids
        }
        if self.use_trained_models:
            for aid, agent in agents.items():
                try:
                    agent.load(f"{self.settings.model_dir}/{aid}")
                except Exception:
                    pass
            for agent in agents.values():
                agent.set_eval_mode()
        self.coordinator = MultiAgentCoordinator(agents)
        self.running = True
        self._obs = None

    def step(self) -> List[Dict[str, Any]]:
        if not self.env or not self.coordinator:
            return []
        if self._obs is None:
            self._obs, _ = self.env.reset()
        actions = self.coordinator.choose_actions(self._obs, explore=False)
        self._obs, rewards, dones, _, infos = self.env.step(actions)
        payload = []
        for aid in self.env.agent_ids:
            info = infos.get(aid, {})
            payload.append(
                {
                    "id": aid,
                    "queue": int(info.get("queue_length", 0)),
                    "wait": float(info.get("waiting_time", 0)),
                    "reward": float(rewards.get(aid, 0)),
                    "throughput": int(info.get("throughput", 0)),
                    "pressure": float(info.get("pressure", 0)),
                    "density": float(info.get("density", 0)),
                    "phase": int(info.get("phase", 0)),
                }
            )
        if any(dones.values()):
            self._obs, _ = self.env.reset()
        return payload

    async def stream(self, websocket, interval: float = 0.1):
        self.initialize()
        try:
            await websocket.send_json({"type": "status", "mode": self.mode})
            while self.running:
                payload = self.step()
                await websocket.send_text(json.dumps(payload))
                await asyncio.sleep(interval)
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        self.running = False
        if self.env:
            self.env.close()
            self.env = None
        self._obs = None
