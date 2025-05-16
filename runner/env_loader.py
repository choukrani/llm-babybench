# runner/env_loader.py

import json
from typing import List, Union

import gymnasium as gym
from gymnasium import Env
from custom.custom_minigrid_envs.register import register_envs 

def make_env(env_name: str, seed: int) -> Env:
    """
    Create an environment from config file specifications.
    """
    register_envs()
    env = gym.make(env_name, tile_size=32, render_mode='rgb_array')
    env.reset(seed=seed)
    return env

def load_envs(config: dict) -> List:
    """
    Load a list of environments either from file or dynamically generated.
    """
    envs = []
    env_config = config.get("env", {})

    env = make_env(env_config["env_name"], env_config.get("seed", 0))
    envs.append(env)

    return envs
