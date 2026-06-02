# System Architecture

## Overview

The platform separates concerns into five layers: simulation, learning, orchestration, API, and presentation.

## Simulation Layer

- `MultiAgentSumoEnv`: Production environment backed by SUMO TraCI
- `MockTrafficEnv`: Stochastic fallback for development without SUMO
- `TrafficObserver`: State normalization and pressure metrics
- `RewardShaper`: Local + cooperative + fairness reward composition
- `IntersectionGraph`: Spatial neighbor topology for cooperative rewards

## Learning Layer

| Module | Responsibility |
|--------|----------------|
| `agents/` | Algorithm implementations (DQN, Rainbow, PPO, QMIX, MADDPG) |
| `agents/networks.py` | Neural architectures (Dueling, Noisy, Mixing, PPO) |
| `agents/replay_buffer.py` | Experience replay variants |
| `training/trainer.py` | Episode loop, DB logging, checkpointing |
| `training/callbacks.py` | Hooks for metrics, TensorBoard, checkpoints |
| `evaluation/` | Benchmarks and trained policy evaluation |

## Orchestration Layer

- `core/registry.py`: Plugin registry for algorithms and environments
- `core/config.py`: Centralized Pydantic settings
- `core/env_loader.py`: Auto-select SUMO or mock environment
- `bootstrap.py`: Eager registration of plugins at import time

## API Layer

- FastAPI application with modular routers
- `SimulationService`: WebSocket streaming engine
- `TrainingService`: Background thread training jobs
- `AnalyticsService`: SQL aggregation queries
- `ModelService`: Checkpoint and policy artifact listing

## Data Flow

1. Environment produces per-agent observations and cooperative rewards
2. Agents select discrete phase actions via epsilon-greedy or policy sampling
3. Transitions stored in replay buffers; gradients applied each step
4. Metrics persisted to SQLite; checkpoints saved periodically
5. API exposes aggregated analytics and live WebSocket feeds
6. React dashboard visualizes real-time intersection state

## Deployment Topology

```
Developer Machine / Server
├── Python 3.10+ (training, API)
├── SUMO 1.18+ (optional, simulation)
├── SQLite (metrics)
├── Node.js (frontend build)
└── Docker Compose (optional bundled stack)
```

## Extension Points

- Register new algorithms via `@AlgorithmRegistry.register("name")`
- Register environments via `@EnvironmentRegistry.register("name")`
- Add reward terms in `RewardShaper`
- Add API routes under `backend/routes/`
