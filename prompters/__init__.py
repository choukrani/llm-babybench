# /prompters/__init__.py

from prompters.zero_shot import ZeroShotPrompter
from prompters.few_shot import FewShotPrompter
from prompters.cot import CoTPrompter
from prompters.tot import ToTPrompter

from prompters.utils import Task

def get_prompter(name: str):
    name = name.lower()
    if name == "zero_shot":
        return ZeroShotPrompter()
    elif name == "few_shot":
        return FewShotPrompter()
    elif name == "cot":
        return CoTPrompter()
    elif name == "tot":
        return ToTPrompter()
    else:
        raise ValueError(f"Unknown prompter: {name}")
    