import random
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np


class CircularReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: List = []
        self.position = 0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            np.array(state, dtype=np.float32),
            np.array(action, dtype=np.int64),
            np.array(reward, dtype=np.float32),
            np.array(next_state, dtype=np.float32),
            np.array(done, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class PrioritizedReplayBuffer:
    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.buffer: List = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple:
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[: len(self.buffer)]

        probs = priorities**self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        beta = min(
            1.0,
            self.beta_start
            + self.frame * (1.0 - self.beta_start) / self.beta_frames,
        )
        self.frame += 1

        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()

        batch = [self.buffer[idx] for idx in indices]
        state, action, reward, next_state, done = zip(*batch)

        return (
            np.array(state, dtype=np.float32),
            np.array(action, dtype=np.int64),
            np.array(reward, dtype=np.float32),
            np.array(next_state, dtype=np.float32),
            np.array(done, dtype=np.float32),
            indices,
            np.array(weights, dtype=np.float32),
        )

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-6
            self.max_priority = max(self.max_priority, priority)

    def __len__(self) -> int:
        return len(self.buffer)


class NStepReplayBuffer:
    def __init__(self, capacity: int, n_step: int = 3, gamma: float = 0.99):
        self.capacity = capacity
        self.n_step = n_step
        self.gamma = gamma
        self.buffer: List = []
        self.n_step_buffer: deque = deque(maxlen=n_step)
        self.position = 0

    def _get_n_step_info(self) -> Optional[Tuple]:
        if len(self.n_step_buffer) < self.n_step:
            return None
        reward = 0.0
        for idx, (_, _, r, _, d) in enumerate(self.n_step_buffer):
            reward += (self.gamma**idx) * r
            if d:
                break
        state, action, _, _, _ = self.n_step_buffer[0]
        _, _, _, next_state, done = self.n_step_buffer[-1]
        return state, action, reward, next_state, done

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.n_step_buffer.append((state, action, reward, next_state, done))
        n_step_transition = self._get_n_step_info()
        if n_step_transition is None:
            return
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = n_step_transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            np.array(state, dtype=np.float32),
            np.array(action, dtype=np.int64),
            np.array(reward, dtype=np.float32),
            np.array(next_state, dtype=np.float32),
            np.array(done, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class MultiAgentReplayBuffer:
    def __init__(self, agent_ids: List[str], capacity: int):
        self.agent_ids = agent_ids
        self.buffers = {aid: CircularReplayBuffer(capacity) for aid in agent_ids}
        self.global_buffer: List = []
        self.global_capacity = capacity
        self.global_position = 0

    def push(
        self,
        agent_id: str,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        global_state: Optional[np.ndarray] = None,
        global_next_state: Optional[np.ndarray] = None,
    ) -> None:
        self.buffers[agent_id].push(state, action, reward, next_state, done)
        if global_state is not None and global_next_state is not None:
            if len(self.global_buffer) < self.global_capacity:
                self.global_buffer.append(None)
            self.global_buffer[self.global_position] = (
                global_state,
                {aid: 0 for aid in self.agent_ids},
                {aid: 0.0 for aid in self.agent_ids},
                global_next_state,
                done,
            )
            self.global_position = (self.global_position + 1) % self.global_capacity

    def sample_agent(self, agent_id: str, batch_size: int) -> Tuple:
        return self.buffers[agent_id].sample(batch_size)

    def __len__(self) -> int:
        return min(len(buf) for buf in self.buffers.values())
