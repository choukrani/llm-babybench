# promtpers/zero_shot.py

from typing import Optional

from prompters.base import AbstractPrompter
from prompters.utils import Task, TEMPLATES

class ZeroShotPrompter(AbstractPrompter):
    def prompt(self, description: str, task: Task, str_action_seq: Optional[str] = None) -> str:
        template = TEMPLATES["zero_shot"].get(task)
        if template:
            if task == Task.PREDICT:
                return template.format(description=description, str_action_seq=str_action_seq)
            else:
                return template.format(description=description)
        raise ValueError(f"Unknown task: {task}")
