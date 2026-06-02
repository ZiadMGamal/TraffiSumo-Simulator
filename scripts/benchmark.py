import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bootstrap
from core.logging import setup_logging
from core.utils import save_json
from core.env_loader import load_environment
from evaluation.benchmarks import BenchmarkSuite


def parse_args():
    parser = argparse.ArgumentParser(description="Run baseline traffic benchmarks")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output", type=str, default="evaluation_results/baselines.json")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logging("benchmark")
    env = load_environment(
        "sumo" if args.config else "auto",
        config_file=args.config,
        use_gui=False,
        fallback_mock=True,
    )
    suite = BenchmarkSuite(env)
    results = suite.run_all(episodes_per_strategy=args.episodes)
    save_json(results, args.output)
    logger.info(f"Benchmark results saved to {args.output}")
    logger.info(json.dumps(results, indent=2))
    env.close()


if __name__ == "__main__":
    main()
