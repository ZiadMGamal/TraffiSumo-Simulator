# Cooperative Smart Traffic Management System Using Multi-Agent Reinforcement Learning

A production-grade research and deployment platform for cooperative traffic signal control using Multi-Agent Reinforcement Learning (MARL) integrated with [SUMO](https://eclipse.dev/sumo/) microscopic traffic simulation.

## Features

- **Multi-algorithm MARL stack**: DQN, Rainbow DQN, PPO, QMIX, MADDPG with pluggable registry
- **Cooperative reward shaping**: Local pressure/wait penalties with neighbor-aware cooperative rewards and fairness regularization
- **SUMO integration**: Gymnasium-compatible multi-agent environment with graph-based neighbor topology
- **Training infrastructure**: Checkpointing, TensorBoard logging, metrics tracking, episode callbacks
- **REST API & WebSockets**: FastAPI backend with analytics, training control, and live simulation streaming
- **React dashboard**: Real-time monitoring, analytics, training center, and system health pages
- **Evaluation suite**: Benchmark baselines (fixed-cycle, random, max-pressure) and trained agent evaluation
- **DevOps ready**: Docker, GitHub Actions CI, pytest suite

## Architecture

```
┌─────────────────┐     WebSocket      ┌──────────────────┐
│  React Dashboard│◄──────────────────►│  FastAPI Backend │
└─────────────────┘                    └────────┬─────────┘
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
                    ▼                         ▼                         ▼
            ┌───────────────┐        ┌───────────────┐        ┌───────────────┐
            │ MARL Trainer  │        │ SUMO Environment│       │ SQLite Analytics│
            │ (DQN/PPO/...) │◄──────►│ Multi-Agent Env │       │ Metrics DB      │
            └───────────────┘        └───────────────┘        └───────────────┘
```

## Project Structure

```
├── agents/              # MARL algorithms, networks, replay buffers
├── backend/             # FastAPI application, services, routes
├── core/                # Configuration, logging, registry
├── data/sumo/           # SUMO network, routes, configuration
├── environment/         # SUMO env, observers, reward shaping, wrappers
├── evaluation/          # Evaluator and baseline benchmarks
├── frontend/            # React + Vite dashboard
├── scripts/             # Training, evaluation, benchmark CLI tools
├── tests/               # Unit tests
└── training/            # Trainer, callbacks, checkpoints
```

## Prerequisites

- Python 3.10+
- [SUMO](https://eclipse.dev/sumo/) 1.18+ with `SUMO_HOME` configured
- Node.js 18+ (for frontend)
- CUDA (optional, for GPU training)

## Quick Start

### 1. Clone and configure

```bash
git clone <your-repo-url>
cd cooperative-smart-traffic-marl
cp .env.example .env
```

Set `SUMO_HOME` to your SUMO installation path.

**Without SUMO:** The API and dashboard work in mock simulation mode automatically when `SUMO_HOME` is not set. You can also force mock mode with `ENV_MODE=mock`.

### 2. Install Python dependencies

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Train agents

```bash
python scripts/train.py --algorithm dqn --episodes 100
```

### 4. Start API server

```bash
python -m backend.main
```

Open API docs at `http://localhost:8000/docs`

### 5. Start dashboard

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`

## CLI Reference

| Command | Description |
|---------|-------------|
| `python scripts/train.py --algorithm dqn --episodes 500` | Train MARL agents |
| `python scripts/evaluate.py --episodes 10` | Evaluate trained models |
| `python scripts/benchmark.py --episodes 5` | Run baseline benchmarks |
| `python scripts/export_models.py` | Export models for deployment |
| `python scripts/generate_network.py` | Generate SUMO grid network |
| `python run.py api` | Start API server (shortcut) |
| `python run.py train --algorithm ppo --episodes 200` | Train via run helper |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/system/health` | Health check |
| GET | `/api/analytics/summary` | Aggregated metrics |
| GET | `/api/analytics/leaderboard` | Intersection rankings |
| POST | `/api/training/start` | Start background training |
| GET | `/api/training/status` | Training status |
| WS | `/ws/traffic` | Live simulation stream |

## Algorithms

| Algorithm | Type | Use Case |
|-----------|------|----------|
| DQN | Value-based | Default discrete phase control |
| Rainbow | Value-based | Prioritized, distributional learning |
| PPO | Policy gradient | On-policy stable training |
| QMIX | Value decomposition | Cooperative team Q-learning |
| MADDPG | Actor-critic | Continuous control extensions |

## Docker

```bash
docker compose up --build
```

Backend: `http://localhost:8000`  
Frontend: `http://localhost:5173`

## Testing

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

## Configuration

Key environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `SUMO_HOME` | — | SUMO installation path |
| `ALGORITHM` | `dqn` | Default training algorithm |
| `TOTAL_EPISODES` | `500` | Training episodes |
| `BATCH_SIZE` | `64` | Replay batch size |
| `COOPERATIVE_WEIGHT` | `0.3` | Neighbor reward weight |

## Research Background

This system implements cooperative MARL for urban traffic signal control where each intersection is controlled by an independent agent. Agents observe local queue lengths, waiting times, and signal phases, then select signal phases while receiving rewards shaped by traffic pressure and cooperative signals from neighboring intersections.

## License

MIT License — see [LICENSE](LICENSE).

## Author

**Ziad Mohamed Gamal**
