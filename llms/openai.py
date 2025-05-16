# llms/openai.py

import os
import openai
from llms.base import AbstractLLM

class OpenAI(AbstractLLM):
    
    def __init__(self, model: str = "gpt-4", temperature: float = 0.0, max_tokens: int = 512, system_prompt: str = ""):

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        
        api_key = os.environ.get("OPENAI_API_KEY")

        if not api_key:
            raise ValueError("OpenAI API key must be set as OPENAI_API_KEY environment variable")
            
        self.client = openai.OpenAI(api_key=api_key)
    
    def generate(self, prompt: str) -> str:

        try:

            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")
        