# Copyright (c) Microsoft Corporation.
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
    """从OpenAI RateLimitError中解析等待时间。

    Args:
        error (openai.RateLimitError): OpenAI API的速率限制错误。

    Returns:
        Optional[int]: 建议的等待时间（秒），解析失败时返回None。
    """
    try:
        info_str: str = error.args[0]
        info_dict_str: str = info_str[info_str.find("{"):]
        error_info: dict = json.loads(re.compile(r"(?<!\\)'").sub('"', info_dict_str))
        error_message = error_info["error"]["message"]
        matches = re.search(r"Try again in (\d+) seconds", error_message)
        wait_time = int(matches.group(1)) + 3  # 添加3秒缓冲
        return wait_time
    except Exception:
        return None


class CompanyOpenAIClient(BaseLLMClient):
    """公司内部OpenAI endpoint客户端实现。"""

    NAME = "CompanyOpenAIClient"

    def __init__(
        self,
        location: str = None,
        auto_dump: bool = True,
        logger: Logger = None,
        llm_config: dict = None,
        max_attempt: int = 5,
        exponential_backoff_factor: int = None,
        unit_wait_time: int = 60,
        **kwargs,
    ) -> None:
        """公司内部OpenAI endpoints的LLM通信客户端。

        Args:
            location (str): LLM客户端通信缓存的文件位置。设置为None时不会创建缓存。默认为None。
            auto_dump (bool): 是否自动保存客户端的通信缓存。默认为True。
            logger (Logger): 客户端日志记录器。默认为None。
            llm_config (dict): LLM配置参数，用于兼容工作流程调用。默认为None。
            max_attempt (int): LLM请求的最大尝试次数。达到max_attempt时会跳过请求。默认为5。
            exponential_backoff_factor (int): 设置以启用指数退避重试方式。每次等待时间为
                `exponential_backoff_factor ^ num_attempt`。设置为None以禁用并使用`unit_wait_time`方式。默认为None。
            unit_wait_time (int): 仅在禁用指数退避模式时使用`unit_wait_time`。每次等待时间为
                `unit_wait_time * num_attempt`，时间单位为秒(s)。默认为60。
            **kwargs: OpenAI客户端初始化的额外参数。
            yml配置示例:
            ...
                llm_client:
                    module_path: pikerag.llm_client
                    class_name: CompanyOpenAIClient
                    args:
                        api_key: <your_api_key>
                        base_url: <your_base_url>
                        trace_id: <optional_trace_id>
            ...
        """
        super().__init__(location, auto_dump, logger, max_attempt, exponential_backoff_factor, unit_wait_time, **kwargs)

        # 存储llm_config供后续使用（虽然在初始化时不直接使用，但保持兼容性）
        self._llm_config = llm_config or {}

        # 配置客户端参数
        client_configs = {
            "api_key": kwargs.get("api_key") or os.environ.get("COMPANY_OPENAI_API_KEY"),
            "base_url": kwargs.get("base_url") or os.environ.get("COMPANY_OPENAI_BASE_URL", "https://aigc.sankuai.com/v1/openai/native"),
        }
        
        # 验证必需参数
        if not client_configs["api_key"]:
            raise ValueError("API key is required. Please set 'api_key' parameter or 'COMPANY_OPENAI_API_KEY' environment variable.")
        
        self._client = OpenAI(**client_configs)
        
        # 存储可选的trace_id，用于请求追踪
        self._trace_id = kwargs.get("trace_id") or os.environ.get("COMPANY_OPENAI_TRACE_ID")

    def _get_extra_headers(self) -> dict:
        """构建额外的请求头。
        
        Returns:
            dict: 包含额外头信息的字典。
        """
        headers = {}
        if self._trace_id:
            headers["M-TraceId"] = self._trace_id
        return headers

    def _get_response_with_messages(self, messages: List[dict], **llm_config) -> ChatCompletion:
        """通过重试机制从OpenAI chat completion API获取响应。

        Args:
            messages (List[dict]): 发送到OpenAI chat completion API的消息。
            **llm_config: chat completion API的额外配置。

        Returns:
            ChatCompletion: OpenAI API的响应。
        """
        response: ChatCompletion = None
        num_attempt: int = 0

        # 准备请求参数
        request_params = {
            "messages": messages,
            **llm_config
        }
        
        # 添加额外头信息
        extra_headers = self._get_extra_headers()
        if extra_headers:
            request_params["extra_headers"] = extra_headers

        while num_attempt < self._max_attempt:
            try:
                response = self._client.chat.completions.create(**request_params)
                break
            except openai.RateLimitError as e:
                self.warning("  请求失败，原因：RateLimitError...")
                wait_time = parse_wait_time_from_error(e)
                self._wait(num_attempt, wait_time=wait_time)
                self.warning("  正在重试...")
            except openai.BadRequestError as e:
                self.warning(f"  请求失败，原因：异常: {e}")
                self.warning("  跳过此请求...")
                break
            except Exception as e:
                self.warning(f"  请求失败，原因：异常: {e}")
                num_attempt += 1
                self._wait(num_attempt)
                self.warning("  正在重试...")

        return response

    def _get_content_from_response(self, response: ChatCompletion, messages: List[dict] = None) -> str:
        """从OpenAI chat completion响应中提取内容。

        Args:
            response (ChatCompletion): OpenAI chat completion API的响应。
            messages (List[dict], optional): 发送到API的原始消息。默认为None。

        Returns:
            str: 提取的内容，提取失败时返回空字符串。
        """
        try:
            content = response.choices[0].message.content
            if content is None:
                finish_reason = response.choices[0].finish_reason
                warning_message = f"由于{finish_reason}返回非内容结果"

                if "content_filter" in finish_reason:
                    content_filter_results = getattr(response.choices[0], 'content_filter_results', {})
                    for reason, res_dict in content_filter_results.items():
                        if res_dict.get("filtered") is True or res_dict.get("severity") != "safe":
                            warning_message += f", '{reason}': {res_dict}"

                self.warning(warning_message)
                self.debug(f"  -- 完整响应: {response}")
                if messages is not None and len(messages) >= 1:
                    self.debug(f"  -- 最后一条消息: {messages[-1]}")

                content = ""
        except Exception as e:
            self.warning(f"尝试从响应中获取内容时出现异常:\n  {e}")
            self.debug(
                f"  响应: {response}\n"
                f"  最后一条消息: {messages}"
            )
            content = ""

        return content

    def close(self):
        """关闭OpenAI客户端。"""
        super().close()
        self._client.close()


class CompanyOpenAIEmbedding(Embeddings):
    """公司内部OpenAI embedding客户端。"""

    def __init__(self, **kwargs) -> None:
        """初始化公司内部OpenAI embedding客户端。
        
        Args:
            **kwargs: 配置参数，包括：
                client_config: 客户端配置
                model: embedding模型名称
                cache_config: 缓存配置
        """
        client_config = kwargs.get("client_config", {})
        
        # 配置客户端参数
        client_configs = {
            "api_key": client_config.get("api_key") or os.environ.get("COMPANY_OPENAI_API_KEY"),
            "base_url": client_config.get("base_url") or os.environ.get("COMPANY_OPENAI_BASE_URL", "https://aigc.sankuai.com/v1/openai/native"),
        }
        
        # 验证必需参数
        if not client_configs["api_key"]:
            raise ValueError("API key is required. Please set 'api_key' in client_config or 'COMPANY_OPENAI_API_KEY' environment variable.")
        
        self._client = OpenAI(**client_configs)
        self._model = kwargs.get("model", "text-embedding-3-small")

        # 设置缓存
        cache_config = kwargs.get("cache_config", {})
        cache_location = cache_config.get("location", None)
        if cache_location is not None:
            self._cache: PickleDB = PickleDB(location=cache_location)
        else:
            self._cache = None
            
        # 存储可选的trace_id
        self._trace_id = client_config.get("trace_id") or os.environ.get("COMPANY_OPENAI_TRACE_ID")

    def _save_cache(self, query: str, embedding: List[float]) -> None:
        """将embedding保存到缓存。"""
        if self._cache is None:
            return
        self._cache.set(query, embedding)

    def _get_cache(self, query: str) -> Union[List[float], Literal[False]]:
        """从缓存中检索embedding。"""
        if self._cache is None:
            return False
        return self._cache.get(query)

    def _get_extra_headers(self) -> dict:
        """构建额外的请求头。
        
        Returns:
            dict: 包含额外头信息的字典。
        """
        headers = {}
        if self._trace_id:
            headers["M-TraceId"] = self._trace_id
        return headers

    def _get_response(self, texts: Union[str, List[str]]) -> CreateEmbeddingResponse:
        """从OpenAI API获取embedding响应。"""
        while True:
            try:
                # 准备请求参数
                request_params = {
                    "input": texts,
                    "model": self._model
                }
                
                # 添加额外头信息
                extra_headers = self._get_extra_headers()
                if extra_headers:
                    request_params["extra_headers"] = extra_headers
                
                response = self._client.embeddings.create(**request_params)
                break
            except openai.RateLimitError as e:
                expected_wait = parse_wait_time_from_error(e)
                if expected_wait is not None:
                    print(f"Embedding请求失败，原因：RateLimitError，等待{expected_wait}秒")
                    time.sleep(expected_wait)
                else:
                    print("Embedding请求失败，原因：RateLimitError，等待30秒")
                    time.sleep(30)
            except Exception as e:
                print(f"Embedding请求失败，原因：异常 {e}")
                return None

        return response

    def embed_documents(self, texts: List[str], batch_call: bool = False) -> List[List[float]]:
        """嵌入文档列表。"""
        if batch_call:
            response = self._get_response(texts)
            if response is None:
                return [[]] * len(texts)
            embeddings = [res.embedding for res in response.data]
        else:
            embeddings = [self.embed_query(text) for text in texts]
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """嵌入单个查询。"""
        embedding = self._get_cache(text)
        if embedding is False:
            response = self._get_response(text)
            if response is None:
                return []
            embedding = response.data[0].embedding
            self._save_cache(text, embedding)
        return embedding