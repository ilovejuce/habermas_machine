"""
This module provides a client for interacting with language models via the Poe API.
It replaces the original implementation that used the Google AI Studio (Gemini) API.
"""

from collections.abc import Collection
import os
import time
import openai

# The 'poe-api' library is used to interact with Poe.
# You can install it with: pip install poe-api
from openai import OpenAI
from typing_extensions import override

from habermas_machine.llm_client import base_client
from habermas_machine.llm_client import utils


class PoeClient(base_client.LLMClient):
  """Language Model that uses the Poe API."""

  def __init__(
      self,
      model_name: str,
      *,
      sleep_periodically: bool = False,
  ) -> None:
    """Initializes the instance.

    Args:
      model_name: Which Poe bot to use. This corresponds to the bot's name
        on poe.com, e.g., 'Claude-3-Opus', 'GPT-4', 'Llama-3-70b'.
      sleep_periodically: Sleep between API calls to avoid potential rate limits.
    """
    # The Poe API key is the 'p-b' cookie value from poe.com.
    # It is recommended to store this token in an environment variable.
    try:
        self._api_key = os.environ['POE_API_KEY']
    except KeyError:
        raise EnvironmentError(
            "The 'POE_API_KEY' environment variable is not set. "
            "Please set it to your Poe 'p-b' cookie value."
        )

    self._model_name = model_name
    self._sleep_periodically = sleep_periodically


    # Configuration for periodic sleeping to manage rate limits.
    self._calls_between_sleeping = 10
    self._n_calls = 0

    self._client = openai.OpenAI(
    api_key=os.getenv("POE_API_KEY"), # https://poe.com/api_key
    base_url="https://api.poe.com/v1",
)

  @override
  def sample_text(
      self,
      prompt: str,
      *,
      max_tokens: int = base_client.DEFAULT_MAX_TOKENS,
      terminators: Collection[str] = base_client.DEFAULT_TERMINATORS,
      temperature: float = base_client.DEFAULT_TEMPERATURE,
      timeout: float = base_client.DEFAULT_TIMEOUT_SECONDS,
      seed: int | None = None,
  ) -> str:
    """
    Generates text by sending a prompt to a specified Poe bot.
    """

    # --- Rate Limiting ---
    self._n_calls += 1
    if self._sleep_periodically and (
        self._n_calls % self._calls_between_sleeping == 0):
      print(f'Sleeping for 10 seconds to respect Poe rate limits...')
      time.sleep(10)

    # --- API Call and Response Handling ---
    response_text = ''
    try:
        resp = self._client.chat.completions.create(
            model=self._model_name,  # Name on Poe
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stop=list(terminators) if terminators else None,
            stream=False,
        )
    
        # Extract Model Text Safely
        response_text = ""
        if resp and getattr(resp, "choices", None):
            msg = resp.choices[0].message
            if msg and getattr(msg, "content", None):
                response_text = msg.content
    
    except Exception as e:
        # Catching a broad exception as the library might raise various errors
        # (e.g., connection errors, invalid bot name, token issues).
        print(f"An error occurred with the Poe API call: {e}")
        print(f"Bot: {self._model_name}")
        print(f"Prompt: {prompt}")

    # --- Post-processing ---
    # Since the Poe API doesn't support 'stop_sequences' (terminators),
    # we manually truncate the response at the first occurrence of a terminator.
    return utils.truncate(response_text, delimiters=terminators) if response_text else ""
