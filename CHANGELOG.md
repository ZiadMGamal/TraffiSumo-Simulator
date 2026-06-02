# Changelog

## [2.0.0] - 2026-05-24

### Added

- Multi-algorithm MARL stack (DQN, Rainbow, PPO, QMIX, MADDPG)
- Plugin registry for algorithms and environments
- Mock traffic environment for development without SUMO
- FastAPI modular routes (analytics, training, simulation, models, system)
- WebSocket live traffic and training status streams
- React dashboard with routing (Dashboard, Analytics, Training, System)
- Training infrastructure with checkpoints, callbacks, metrics tracker
- Evaluation suite and baseline benchmarks
- Docker Compose and GitHub Actions CI
- Comprehensive pytest suite
- SUMO network data under `data/sumo/`

### Changed

- Refactored monolithic backend into services and route modules
- Upgraded DQN agent with save/load, eval mode, and registry integration
- Cooperative reward shaping with graph-based neighbors

### Fixed

- Environment auto-fallback when SUMO is unavailable
- Observation dimension consistency (26-dim state vectors)

## [1.0.0] - Initial

- Basic DQN multi-agent SUMO integration
- Simple FastAPI WebSocket and React dashboard
