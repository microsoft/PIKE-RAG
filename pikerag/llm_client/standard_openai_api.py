# Licensed under the MIT license.

import json
import re
import time
from typing import List, Literal, Optional, Union
import os

import openai
from langchain_core.embeddings import Embeddings
from openai import OpenAI
from openai.types import CreateEmbeddingResponse
from openai.types.chat.chat_completion import ChatCompletion
from pickledb import PickleDB

from pikerag.llm_client.base import BaseLLMClient
from pikerag.utils.logger import Logger


def parse_wait_time_from_error(error: openai.RateLimitError) -> Optional[int]:
    """Parse wait time from OpenAI RateLimitError.

    Args:
        error (openai.RateLimitError): The rate limit error from OpenAI API.

    Returns:
        Optional[int]: The suggested wait time in seconds, None if parsing failed.
    """
    try:
        info_str: str = error.args[0]
        info_dict_str: str = info_str[info_str.find("{"):]
        error_info: dict = json.loads(re.compile(r"(?<!\\)'").sub('"', info_dict_str))
        error_message = error_info["error"]["message"]
        matches = re.search(r"Try again in (\d+) seconds", error_message)
        wait_time = int(matches.group(1)) + 3  # Add 3 seconds buffer
        return wait_time
    except Exception:
        return None


class StandardOpenAIClient(BaseLLMClient):
    """Standard OpenAI client implementation for LLM communication."""

    NAME = "StandardOpenAIClient"

    def __init__(
        self,
        location: str = None,
        auto_dump: bool = True,
        logger: Logger = None,
        max_attempt: int = 5,
        exponential_backoff_factor: int = None,
        unit_wait_time: int = 60,
        **kwargs,
    ) -> None:
        """LLM Communication Client for Standard OpenAI endpoints.

        Args:
            location (str): The file location of the LLM client communication cache. No cache would be created if set to
                None. Defaults to None.
            auto_dump (bool): Automatically save the Client's communication cache or not. Defaults to True.
            logger (Logger): Client logger. Defaults to None.
            max_attempt (int): Maximum attempt time for LLM requesting. Request would be skipped if max_attempt reached.
                Defaults to 5.
            exponential_backoff_factor (int): Set to enable exponential backoff retry manner. Every time the wait time
                would be `exponential_backoff_factor ^ num_attempt`. Set to None to disable and use the `unit_wait_time`
                manner. Defaults to None.
            unit_wait_time (int): `unit_wait_time` would be used only if the exponential backoff mode is disabled. Every
                time the wait time would be `unit_wait_time * num_attempt`, with seconds (s) as the time unit. Defaults
                to 60.
            **kwargs: Additional arguments for OpenAI client initialization.
            yml config example:
            ...
                llm_client:
                    module_path: pikerag.llm_client
                    class_name: StandardOpenAIClient
                    args:{
                        api_key: <your_api_key>
                        base_url: <your_base_url>
                    }
            ...
        """
        super().__init__(location, auto_dump, logger, max_attempt, exponential_backoff_factor, unit_wait_time, **kwargs)

        client_configs = {
            "api_key": kwargs.get("api_key") or os.environ.get("OPENAI_API_KEY"),
            "base_url": kwargs.get("base_url") or os.environ.get("OPENAI_BASE_URL"),
        }
        self._client = OpenAI(**client_configs)

    def _get_response_with_messages(self, messages: List[dict], **llm_config) -> ChatCompletion:
        """Get response from OpenAI chat completion API with retry mechanism.

        Args:
            messages (List[dict]): The messages to send to OpenAI chat completion API.
            **llm_config: Additional configuration for the chat completion API.

        Returns:
            ChatCompletion: The response from OpenAI API.
        """
        response: ChatCompletion = None
        num_attempt: int = 0

        while num_attempt < self._max_attempt:
            try:
                response = self._client.chat.completions.create(messages=messages, **llm_config)
                break
            except openai.RateLimitError as e:
                self.warning("  Failed due to RateLimitError...")
                wait_time = parse_wait_time_from_error(e)
                self._wait(num_attempt, wait_time=wait_time)
                self.warning("  Retrying...")
            except openai.BadRequestError as e:
                self.warning(f"  Failed due to Exception: {e}")
                self.warning("  Skip this request...")
                break
            except Exception as e:
                self.warning(f"  Failed due to Exception: {e}")
                num_attempt += 1
                self._wait(num_attempt)
                self.warning("  Retrying...")

        return response

    def _get_content_from_response(self, response: ChatCompletion, messages: List[dict] = None) -> str:
        """Extract content from OpenAI chat completion response.

        Args:
            response (ChatCompletion): The response from OpenAI chat completion API.
            messages (List[dict], optional): The original messages sent to API. Defaults to None.

        Returns:
            str: The extracted content or empty string if extraction failed.
        """
        try:
            content = response.choices[0].message.content
            if content is None:
                finish_reason = response.choices[0].finish_reason
                warning_message = f"Non-Content returned due to {finish_reason}"

                if "content_filter" in finish_reason:
                    for reason, res_dict in response.choices[0].content_filter_results.items():
                        if res_dict["filtered"] is True or res_dict["severity"] != "safe":
                            warning_message += f", '{reason}': {res_dict}"

                self.warning(warning_message)
                self.debug(f"  -- Complete response: {response}")
                if messages is not None and len(messages) >= 1:
                    self.debug(f"  -- Last message: {messages[-1]}")

                content = ""
        except Exception as e:
            self.warning(f"Try to get content from response but get exception:\n  {e}")
            self.debug(
                f"  Response: {response}\n"
                f"  Last message: {messages}"
            )
            content = ""

        return content

    def close(self):
        """Close the OpenAI client."""
        super().close()
        self._client.close()


class StandardOpenAIEmbedding(Embeddings):
    """Standard OpenAI embedding client."""

    def __init__(self, **kwargs) -> None:
        client_configs = kwargs.get("client_config", {})
        self._client = OpenAI(**client_configs)
        self._model = kwargs.get("model", "text-embedding-ada-002")

        cache_config = kwargs.get("cache_config", {})
        cache_location = cache_config.get("location", None)
        if cache_location is not None:
            self._cache: PickleDB = PickleDB(location=cache_location)
        else:
            self._cache = None

    def _save_cache(self, query: str, embedding: List[float]) -> None:
        """Save embedding to cache."""
        if self._cache is None:
            return
        self._cache.set(query, embedding)

    def _get_cache(self, query: str) -> Union[List[float], Literal[False]]:
        """Retrieve embedding from cache."""
        if self._cache is None:
            return False
        return self._cache.get(query)

    def _get_response(self, texts: Union[str, List[str]]) -> CreateEmbeddingResponse:
        """Get embedding response from OpenAI API."""
        while True:
            try:
                response = self._client.embeddings.create(input=texts, model=self._model)
                break
            except openai.RateLimitError as e:
                expected_wait = parse_wait_time_from_error(e)
                if expected_wait is not None:
                    print(f"Embedding failed due to RateLimitError, wait for {expected_wait} seconds")
                    time.sleep(expected_wait)
                else:
                    print("Embedding failed due to RateLimitError, wait for 30 seconds")
                    time.sleep(30)
            except Exception as e:
                print(f"Embedding failed due to exception {e}")
                return None

        return response

    def embed_documents(self, texts: List[str], batch_call: bool = False) -> List[List[float]]:
        """Embed a list of documents."""
        if batch_call:
            response = self._get_response(texts)
            embeddings = [res.embedding for res in response.data]
        else:
            embeddings = [self.embed_query(text) for text in texts]
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        embedding = self._get_cache(text)
        if embedding is False:
            response = self._get_response(text)
            embedding = response.data[0].embedding
            self._save_cache(text, embedding)
        return embedding