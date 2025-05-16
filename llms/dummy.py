# llms/dummy.py

from llms.base import AbstractLLM

class DummyLLM(AbstractLLM):
    
    def generate(self, prompt: str) -> str:
        return "LLM Output"
        