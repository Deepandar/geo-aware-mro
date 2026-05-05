# src/utils/config_loader.py

import os
import yaml


def load_config():
    env = os.getenv("ENV", "dev")
    path = f"config/env/{env}.yaml"

    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path, "r") as f:
        return yaml.safe_load(f)["pipeline"]
