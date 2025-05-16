# promtpers/base.py

from typing import Optional

from abc import ABC, abstractmethod
from prompters.utils import Task

class AbstractPrompter(ABC):
    @abstractmethod
    def prompt(self, description: str, task: Task, str_action_seq: Optional[str] = None) -> str:
        """Return a prompt to feed into the LLM for a specific BabyAI task."""
        pass
    