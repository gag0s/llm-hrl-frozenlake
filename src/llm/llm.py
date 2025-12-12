from openai import OpenAI
from dotenv import load_dotenv
import os


class BaseLLM:
    """Abstract interface for an LLM backend used by the PromptBox."""
    def get_response(self, prompt: str) -> str:
        raise NotImplementedError


class MockLLM(BaseLLM):
    """Deterministic stand-in for development/testing (no API calls)."""
    def get_response(self, prompt: str) -> str:
        response = "[56, 50, 45, 47, 63]"
        print("MockLLM response: " + response)
        return response


class ChatGPTLLM(BaseLLM):
    """
    OpenAI Chat Completions client used to request subgoal lists.
    Expects OPENAI_API_KEY in environment or .env file.
    """
    def __init__(self):
        load_dotenv()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-5"

    def get_response(self, prompt: str) -> str:
        reply = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a precise and structured assistant."},
                {"role": "user", "content": prompt},
            ],
        )

        response = reply.choices[0].message.content.strip()
        print("ChatGPT: " + response)
        return response
