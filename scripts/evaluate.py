import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bootstrap
from core.logging import setup_logging
from evaluation.evaluator import MARLEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained MARL agents")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model-dir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logging("evaluate")
    evaluator = MARLEvaluator(
        config_file=args.config,
        model_dir=args.model_dir,
        use_gui=args.gui,
    )
    if args.episodes == 1 and args.gui:
        result = evaluator.run_episode()
        logger.info(f"Episode result: {result}")
    else:
        summary = evaluator.run_benchmark(num_episodes=args.episodes)
        logger.info(f"Benchmark summary: {summary}")
    evaluator.close()


if __name__ == "__main__":
    main()
