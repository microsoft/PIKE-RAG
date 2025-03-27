import json
import os
import re
import time
from typing import Callable, List, Literal, Optional, Union

from langchain_core.embeddings import Embeddings
from ollama import Client as OllamaClient # Assuming an equivalent Ollama client exists
from pickledb import PickleDB

from pikerag.llm_client.base import BaseLLMClient
from pikerag.utils.logger import Logger

def parse_wait_time_from_error(error: Exception) -> Optional[int]:
    try:
        info_str: str = str(error)
        matches = re.search(r"Try again in (\d+) seconds", info_str)
        if matches:
            return int(matches.group(1)) + 3  # Wait an additional 3 seconds
    except Exception:
        pass
    return None

class OllamaLLMClient(BaseLLMClient):
    NAME = "OllamaLLMClient"

    def __init__(
        self, location: str = None, auto_dump: bool = True, logger: Logger = None,
        max_attempt: int = 5, exponential_backoff_factor: int = None, unit_wait_time: int = 60, **kwargs,
    ) -> None:
        super().__init__(location, auto_dump, logger, max_attempt, exponential_backoff_factor, unit_wait_time, **kwargs)

        # client_configs = kwargs.get("client_config", {})
        # print(client_configs.get("OLLAMA_HOST", None) )
        # if client_configs.get("OLLAMA_HOST", None) is None and os.environ.get("OLLAMA_HOST", None) is None:
        #     client_configs["base_url"] = "http://10.5.108.210:11434"
        base_url = os.environ.get("OLLAMA_HOST")

        self._client = OllamaClient(**kwargs.get("client_config", {}))

        #self._client = OllamaClient(**kwargs.get("client_config", {}))

    def _get_response_with_messages(self, messages: List[dict], **llm_config) -> dict:
        response = None
        num_attempt = 0
        while num_attempt < self._max_attempt:
            try:
                response = self._client.chat(messages=messages, **llm_config)
                break
            except Exception as e:
                self.warning(f"Request failed due to: {e}")
                num_attempt += 1
                wait_time = parse_wait_time_from_error(e) or (self._unit_wait_time * num_attempt)
                time.sleep(wait_time)
                self.warning("Retrying...")
        return response

    def _get_content_from_response(self, response: dict, messages: List[dict] = None) -> str:
        try:
            resp = response.get("message", {}).get("content", "")
            print(resp)
            #resp = response.get("message", {}).get("content", "")[0]
            return resp
        except Exception as e:
            self.warning(f"Error extracting content: {e}")
            return ""

    def close(self):
        super().close()

class OllamaEmbedding(Embeddings):
    def __init__(self, **kwargs) -> None:
        client_configs = kwargs.get("client_config", {})
        base_url = client_configs.get("OLLAMA_URL")
        #model = client_configs.get("OLLAMA_MODEL")
        embed_model = client_configs.get("OLLAMA_EMBED_MODEL")

        self._client = OllamaClient(base_url=base_url, model=embed_model)
        self._model = kwargs.get("model", "nomic-embed-text:latest")
        cache_config = kwargs.get("cache_config", {})
        self._cache = PickleDB(location=cache_config.get("location")) if cache_config.get("location") else None

    def _save_cache(self, query: str, embedding: List[float]) -> None:
        if self._cache:
            self._cache.set(query, embedding)

    def _get_cache(self, query: str) -> Union[List[float], Literal[False]]:
        return self._cache.get(query) if self._cache else False

    def _get_response(self, texts: Union[str, List[str]]) -> dict:
        while True:
            try:
                return self._client.embeddings(input=texts, model=self._model)
            except Exception as e:
                wait_time = parse_wait_time_from_error(e) or 30
                self.warning(f"Embedding request failed: {e}, waiting {wait_time} seconds...")
                time.sleep(wait_time)

    def embed_documents(self, texts: List[str], batch_call: bool = False) -> List[List[float]]:
        if batch_call:
            response = self._get_response(texts)
            return [res["embedding"] for res in response["data"]]
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        embedding = self._get_cache(text)
        if embedding is False:
            response = self._get_response(text)
            embedding = response["data"][0]["embedding"]
            self._save_cache(text, embedding)
        return embedding
