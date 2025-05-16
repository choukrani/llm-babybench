# processors/base.py

from abc import ABC, abstractmethod
from typing import Any, Optional


class EnvProcessor(ABC):

    @abstractmethod
    def process(self, env_name: str, seed: int, task: Optional[str] = None, max_steps: int = 1000) -> Any:
        pass
    