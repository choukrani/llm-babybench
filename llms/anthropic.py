# llms/anthropic.py

import os
import anthropic
from llms.base import AbstractLLM

class Anthropic(AbstractLLM):

    def __init__(self, model: str = "claude-3-7-sonnet-20250219", temperature: float = 0.0, max_tokens: int = 512, system_prompt: str = ""):

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        
        if not api_key:
            raise ValueError("Anthropic API key must be set as ANTHROPIC_API_KEY environment variable")
        
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def generate(self, prompt: str) -> str:
    
        try:

            messages = [{"role": "user", "content": prompt}]
            
            response = self.client.messages.create(
                model=self.model,
                messages=messages,
                system=self.system_prompt if self.system_prompt else "",
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return response.content[0].text.strip()
            
        except Exception as e:
            raise Exception(f"Anthropic API error: {str(e)}")
            