# runner/config_loader.py

import yaml
from pathlib import Path


REQUIRED_FIELDS = ["env", "inputs", "eval", "llm"]#, "output"]


def load_config(config_path: str) -> dict:
    """
    Load and validate a YAML configuration file.
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    _validate_config(config)
    return config


def _validate_config(config: dict):
    """
    Ensure required fields exist in the config.
    """
    for field in REQUIRED_FIELDS:
        if field not in config:
            raise ValueError(f"Missing required field in config: '{field}'")

    # Nested checks for subfields
    if "env" in config:
        if "env" not in config["env"] or "seed" not in config["env"]:
            raise ValueError("Config 'env' must include both 'env' and 'seed'.")

    if "inputs" in config:
        for key in ["formatter", "processor", "prompter"]:
            if key not in config["inputs"]:
                raise ValueError(f"Missing input field: '{key}'")

    if "eval" in config:
        for key in ["task", "evaluator"]:
            if key not in config["eval"]:
                raise ValueError(f"Missing eval field: '{key}'")

    if "llm" in config:
        for key in ["name", "model"]:
            if key not in config["llm"]:
                raise ValueError(f"Missing LLM field: '{key}'")
