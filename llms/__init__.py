# llms/__init__.py

from llms.dummy import DummyLLM
from llms.openai import OpenAI
from llms.anthropic import Anthropic
from llms.deepinfra import DeepInfra

def get_llm(name: str, model_name: str = None, **kwargs):
    
    if name == "dummy":
        return DummyLLM()
    elif name == "openai":
        return OpenAI(**kwargs)
    elif name == "anthropic":
        return Anthropic(**kwargs)
    elif name == "deepinfra":
        return DeepInfra(**kwargs)
    else:
        raise ValueError(f"Unknown LLM name: {name}")
    