# backend/llm/client.py
import os
import httpx
from dotenv import load_dotenv

load_dotenv()

class LLMClient:
    def __init__(self):
        self.base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1").rstrip("/")
        self.model = os.getenv("LLM_MODEL", "llama3.1:8b")

    def chat(self, messages, temperature=0.2, timeout=90) -> str:
        """
        OpenAI-compatible endpoint:
        POST {base_url}/chat/completions
        Return: choices[0].message.content
        """
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": False
        }

        try:
            with httpx.Client(timeout=timeout) as client:
                r = client.post(url, json=payload)
                r.raise_for_status()
                data = r.json()
        except httpx.HTTPError as e:
            raise RuntimeError(f"LLM HTTP error: {e}") from e

        try:
            return data["choices"][0]["message"]["content"]
        except Exception:
            raise RuntimeError(f"Unexpected LLM response schema: {data}")
