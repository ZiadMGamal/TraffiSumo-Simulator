import os
import sys
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from core.config import get_settings
from core.registry import EnvironmentRegistry
from environment.graph_topology import IntersectionGraph
from environment.observers import TrafficObserver
from environment.reward_shaping import RewardShaper

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))

try:
    import traci
except ImportError:
    traci = None


@EnvironmentRegistry.register("sumo")
class MultiAgentSumoEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        config_file: Optional[str] = None,
        use_gui: bool = False,
        max_steps: int = 3600,
        reward_shaper: Optional[RewardShaper] = None,
    ):
        super().__init__()
        settings = get_settings()
        self.config_file = config_file or str(settings.sumo_config_path)
        self.use_gui = use_gui
        self.max_steps = max_steps
        self.current_step = 0
        self.sumo_binary = "sumo-gui" if self.use_gui else "sumo"
        self.observer = TrafficObserver(max_lanes=settings.max_lanes)
        self.reward_shaper = reward_shaper or RewardShaper(
            cooperative_weight=settings.cooperative_weight,
            local_weight=settings.local_reward_weight,
        )
        self.graph = IntersectionGraph()
        self.agent_ids: list = []
        self.intersection_data: Dict[str, Dict[str, Any]] = {}
        self._initialize_network()

    def _initialize_network(self) -> None:
        settings = get_settings()
        if traci is None:
            raise RuntimeError("SUMO traci module not available. Set SUMO_HOME.")
        traci.start(
            [self.sumo_binary, "-c", self.config_file, "--start", "--quit-on-end"]
        )
        self.agent_ids = list(traci.trafficlight.getIDList())
        for tl_id in self.agent_ids:
            controlled_lanes = list(set(traci.trafficlight.getControlledLanes(tl_id)))
            links = traci.trafficlight.getControlledLinks(tl_id)
            outgoing_lanes = list(
                set([link[0][1] for link in links if link and link[0]])
            )
            self.intersection_data[tl_id] = {
                "incoming": controlled_lanes,
                "outgoing": outgoing_lanes,
                "last_phase": 0,
                "phase_duration": 0,
                "total_throughput": 0,
            }
        self.graph.build_from_sumo(traci)
        obs_dim = settings.max_lanes * 2 + 2
        self.action_spaces = {aid: spaces.Discrete(4) for aid in self.agent_ids}
        self.observation_spaces = {
            aid: spaces.Box(low=0, high=1, shape=(obs_dim,), dtype=np.float32)
            for aid in self.agent_ids
        }
        traci.close()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        if traci.isLoaded():
            traci.close()
        traci.start(
            [
                self.sumo_binary,
                "-c",
                self.config_file,
                "--waiting-time-memory",
                "3600",
                "--no-warnings",
                "true",
            ]
        )
        for aid in self.agent_ids:
            self.intersection_data[aid]["last_phase"] = 0
            self.intersection_data[aid]["phase_duration"] = 0
            self.intersection_data[aid]["total_throughput"] = 0
        self.reward_shaper.reset()
        return self._get_observations(), {"agent_ids": self.agent_ids}

    def step(self, actions: Dict[str, int]):
        self.current_step += 1
        for aid, action in actions.items():
            if aid not in self.intersection_data:
                continue
            curr_phase = traci.trafficlight.getPhase(aid)
            target_phase = int(action) * 2
            if curr_phase != target_phase:
                traci.trafficlight.setPhase(aid, curr_phase + 1)
                self.intersection_data[aid]["phase_duration"] = 0
            else:
                traci.trafficlight.setPhase(aid, target_phase)
                self.intersection_data[aid]["phase_duration"] += 1
        traci.simulationStep()
        obs = self._get_observations()
        rewards = self._calculate_rewards()
        terminated = self.current_step >= self.max_steps
        dones = {aid: terminated for aid in self.agent_ids}
        infos = {aid: self._get_metrics(aid) for aid in self.agent_ids}
        infos["global_state"] = self.observer.build_global_state(obs)
        return obs, rewards, dones, False, infos

    def _get_observations(self) -> Dict[str, np.ndarray]:
        obs = {}
        for aid in self.agent_ids:
            data = self.intersection_data[aid]
            queues = [
                traci.lane.getLastStepHaltingNumber(l) for l in data["incoming"]
            ]
            waits = [traci.lane.getWaitingTime(l) for l in data["incoming"]]
            speeds = [
                traci.lane.getLastStepMeanSpeed(l) for l in data["incoming"]
            ]
            curr_phase = traci.trafficlight.getPhase(aid)
            state = self.observer.normalize_state(
                queues, waits, curr_phase, data["phase_duration"], speeds
            )
            self.observer.record(aid, state)
            obs[aid] = state[: self.observation_spaces[aid].shape[0]]
        return obs

    def _calculate_rewards(self) -> Dict[str, float]:
        local_rewards = {}
        queue_lengths = {}
        for aid in self.agent_ids:
            data = self.intersection_data[aid]
            pressure = self.observer.compute_pressure(
                data["incoming"], data["outgoing"], traci
            )
            total_wait = sum(traci.lane.getWaitingTime(l) for l in data["incoming"])
            arrived = traci.simulation.getArrivedNumber()
            throughput_delta = arrived - data.get("last_arrived", 0)
            data["last_arrived"] = arrived
            local_rewards[aid] = self.reward_shaper.compute_local_reward(
                aid, pressure, total_wait, throughput_delta
            )
            queue_lengths[aid] = sum(
                traci.lane.getLastStepHaltingNumber(l) for l in data["incoming"]
            )
        cooperative = self.reward_shaper.compute_cooperative_rewards(
            local_rewards, self.graph.to_dict()
        )
        return self.reward_shaper.apply_fairness_penalty(cooperative, queue_lengths)

    def _get_metrics(self, aid: str) -> Dict[str, float]:
        data = self.intersection_data[aid]
        return {
            "queue_length": float(
                sum(traci.lane.getLastStepHaltingNumber(l) for l in data["incoming"])
            ),
            "waiting_time": float(
                sum(traci.lane.getWaitingTime(l) for l in data["incoming"])
            ),
            "throughput": int(traci.simulation.getArrivedNumber()),
            "pressure": self.observer.compute_pressure(
                data["incoming"], data["outgoing"], traci
            ),
            "density": self.observer.compute_density(data["incoming"], traci),
            "phase": int(traci.trafficlight.getPhase(aid)),
            "phase_duration": int(data["phase_duration"]),
        }

    def get_neighbor_map(self) -> Dict[str, list]:
        return self.graph.to_dict()

    def close(self):
        try:
            if traci and traci.isLoaded():
                traci.close()
        except Exception:
            pass
