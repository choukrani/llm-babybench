# llms/deepinfra.py

import os
from openai import OpenAI

from llms.base import AbstractLLM


class DeepInfra(AbstractLLM):
    
    def __init__(self, model: str = "meta-llama/Meta-Llama-3-70B-Instruct", temperature: float = 0.0, max_tokens: int = 512, system_prompt: str = ""):

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        
        api_key = os.environ.get("DEEPINFRA_API_KEY")

        if not api_key:
            raise ValueError("DeepInfra API key must be set as DEEPINFRA_API_KEY environment variable")
        
        self.api_key = api_key

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepinfra.com/v1/openai",
        )

    
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
        