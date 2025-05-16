# formatters/base.py

from abc import ABC, abstractmethod

import gymnasium as gym
from gymnasium import Env

class EnvFormatter(ABC):
    @abstractmethod
    def format(self, env: Env) -> str:
        pass
    