from typing import Dict, List, Optional

import numpy as np


class TrafficObserver:
    def __init__(self, max_lanes: int = 12):
        self.max_lanes = max_lanes
        self.history: Dict[str, List[np.ndarray]] = {}

    def normalize_state(
        self,
        queues: List[float],
        wait_times: List[float],
        current_phase: int,
        phase_duration: int,
        speeds: Optional[List[float]] = None,
    ) -> np.ndarray:
        queues_norm = np.array(queues, dtype=np.float32) / 50.0
        waits_norm = np.array(wait_times, dtype=np.float32) / 300.0
        padded_queues = np.zeros(self.max_lanes, dtype=np.float32)
        padded_waits = np.zeros(self.max_lanes, dtype=np.float32)
        padded_queues[: min(len(queues_norm), self.max_lanes)] = queues_norm[
            : self.max_lanes
        ]
        padded_waits[: min(len(waits_norm), self.max_lanes)] = waits_norm[
            : self.max_lanes
        ]
        phase_norm = np.array(
            [current_phase / 8.0, phase_duration / 60.0], dtype=np.float32
        )
        components = [padded_queues, padded_waits, phase_norm]
        if speeds is not None:
            speeds_norm = np.array(speeds, dtype=np.float32) / 20.0
            padded_speeds = np.zeros(self.max_lanes, dtype=np.float32)
            padded_speeds[: min(len(speeds_norm), self.max_lanes)] = speeds_norm[
                : self.max_lanes
            ]
            components.append(padded_speeds)
        return np.concatenate(components)

    def compute_pressure(
        self, incoming_lanes: List[str], outgoing_lanes: List[str], traci_module
    ) -> float:
        incoming_count = sum(
            traci_module.lane.getLastStepHaltingNumber(l) for l in incoming_lanes
        )
        outgoing_count = sum(
            traci_module.lane.getLastStepHaltingNumber(l) for l in outgoing_lanes
        )
        return float(incoming_count - outgoing_count)

    def compute_density(self, lanes: List[str], traci_module) -> float:
        total_vehicles = sum(
            traci_module.lane.getLastStepVehicleNumber(l) for l in lanes
        )
        total_length = sum(traci_module.lane.getLength(l) for l in lanes)
        if total_length <= 0:
            return 0.0
        return total_vehicles / total_length

    def build_global_state(self, local_states: Dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate([local_states[aid] for aid in sorted(local_states)])

    def record(self, agent_id: str, state: np.ndarray) -> None:
        if agent_id not in self.history:
            self.history[agent_id] = []
        self.history[agent_id].append(state.copy())
        if len(self.history[agent_id]) > 1000:
            self.history[agent_id] = self.history[agent_id][-500:]
