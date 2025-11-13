"""
This module provides a client for interacting with language models via the Poe API.
It replaces the original implementation that used the Google AI Studio (Gemini) API.
"""

from collections.abc import Collection
import os
import time

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

    Note: The underlying 'poe-api' library does not support several parameters
    from the base client's interface. These parameters are ignored.
    """
    # --- Parameter Handling ---
    # The Poe API, as accessed by the 'poe-api' library, does not directly
    # support the following parameters. They are included for interface
    # compatibility but are not used in the API call.
    del max_tokens  # Not supported. Length is controlled by the model's output.
    del temperature # Not supported. Temperature is configured on the Poe bot itself.
    del timeout     # Not supported. Timeout is handled by the underlying HTTP library.
    del seed        # Not supported. Poe does not offer a seed for reproducibility.

    # --- Rate Limiting ---
    self._n_calls += 1
    if self._sleep_periodically and (
        self._n_calls % self._calls_between_sleeping == 0):
      print(f'Sleeping for 10 seconds to respect Poe rate limits...')
      time.sleep(10)

    # --- API Call and Response Handling ---
    response_text = ''
    try:
      response_text = self._client.chat.completions.create(
        model=self._model_name # Name on Poe
        messages=[
          {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": prompt},
      ],
)
      # The poe-api library streams the response in chunks.
      # We concatenate the 'text_new' part of each chunk to build the full response.
      # 'with_chat_break=True' ensures that each prompt starts a new, clean conversation
      # context, which is suitable for independent sampling tasks.

    except Exception as e:
      # Catching a broad exception as the library might raise various errors
      # (e.g., connection errors, invalid bot name, token issues).
      print(f'An error occurred with the Poe API call: {e}')
      print(f'Bot: {self._model_name}')
      print(f'Prompt: {prompt}')
      return ''  # Return an empty string on failure, maintaining original behavior.

    # --- Post-processing ---
    # Since the Poe API doesn't support 'stop_sequences' (terminators),
    # we manually truncate the response at the first occurrence of a terminator.
    return utils.truncate(response_text, delimiters=terminators)
