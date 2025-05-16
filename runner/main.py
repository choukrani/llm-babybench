# runner/main.py

import argparse
import yaml
from runner.pipeline import run_pipeline


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Run LLM evaluation pipeline from config.")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config YAML file."
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_pipeline(config)


if __name__ == "__main__":
    main()
    