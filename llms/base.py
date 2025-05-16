# llms/base.py

from abc import ABC, abstractmethod

class AbstractLLM(ABC):
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass
