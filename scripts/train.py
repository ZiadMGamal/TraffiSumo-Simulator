import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bootstrap
from core.config import get_settings
from core.logging import setup_logging
from training.trainer import MARLTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train MARL traffic agents")
    parser.add_argument("--algorithm", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    settings = get_settings()
    logger = setup_logging("train")
    trainer = MARLTrainer(
        algorithm=args.algorithm or settings.algorithm,
        config_file=args.config,
        seed=args.seed,
    )
    episodes = args.episodes or settings.total_episodes
    logger.info(f"Training {episodes} episodes with {trainer.algorithm}")
    summary = trainer.train(total_episodes=episodes)
    logger.info(f"Training complete: {summary}")


if __name__ == "__main__":
    main()
