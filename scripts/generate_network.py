import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate SUMO network with netgenerate")
    parser.add_argument("--grid", action="store_true", default=True)
    parser.add_argument("--size", type=int, default=2)
    parser.add_argument("--output-dir", type=str, default="data/sumo")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    net_file = output_dir / "generated.net.xml"
    cmd = [
        "netgenerate",
        "--grid",
        "--grid.x-number",
        str(args.size),
        "--grid.y-number",
        str(args.size),
        "--grid.length",
        "500",
        "--output-file",
        str(net_file),
        "--tls.guess",
        "true",
    ]
    try:
        subprocess.run(cmd, check=True)
        print(f"Generated network: {net_file}")
    except FileNotFoundError:
        print("netgenerate not found. Install SUMO and add to PATH.", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"netgenerate failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
