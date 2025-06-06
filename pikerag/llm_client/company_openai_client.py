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

        # 记录请求开始时间
        request_start_time = time.time()

        # 准备请求参数
        request_params = {
            "messages": messages,
            **llm_config
        }
        
        # 添加额外头信息
        extra_headers = self._get_extra_headers()
        if extra_headers:
            request_params["extra_headers"] = extra_headers

        # 记录请求详细信息
        self.info(f"[Chat请求] 开始发送chat completion请求")
        self.debug(f"[Chat请求] 模型: {request_params.get('model', 'default')}")
        self.debug(f"[Chat请求] 消息数量: {len(messages)}")
        self.debug(f"[Chat请求] 温度: {request_params.get('temperature', 'default')}")
        self.debug(f"[Chat请求] 最大tokens: {request_params.get('max_tokens', 'default')}")
        if self._trace_id:
            self.debug(f"[Chat请求] Trace ID: {self._trace_id}")
        
        # 记录最后一条用户消息的摘要（避免记录过长的内容）
        if messages and len(messages) > 0:
            last_message = messages[-1]
            content_preview = str(last_message.get('content', ''))[:200]
            if len(str(last_message.get('content', ''))) > 200:
                content_preview += "..."
            self.debug(f"[Chat请求] 最后消息角色: {last_message.get('role', 'unknown')}")
            self.debug(f"[Chat请求] 最后消息内容预览: {content_preview}")

        while num_attempt < self._max_attempt:
            try:
                # 记录当前尝试次数
                if num_attempt > 0:
                    self.info(f"[Chat请求] 第 {num_attempt + 1} 次尝试")
                
                response = self._client.chat.completions.create(**request_params)
                
                # 记录成功响应信息
                request_duration = time.time() - request_start_time
                self.info(f"[Chat请求] 请求成功完成，耗时: {request_duration:.2f}秒")
                
                if response:
                    self.debug(f"[Chat响应] 响应ID: {response.id}")
                    self.debug(f"[Chat响应] 模型: {response.model}")
                    self.debug(f"[Chat响应] 创建时间: {response.created}")
                    
                    if response.usage:
                        self.info(f"[Chat响应] Token使用情况 - 输入: {response.usage.prompt_tokens}, "
                                f"输出: {response.usage.completion_tokens}, "
                                f"总计: {response.usage.total_tokens}")
                    
                    if response.choices and len(response.choices) > 0:
                        choice = response.choices[0]
                        self.debug(f"[Chat响应] 结束原因: {choice.finish_reason}")
                        
                        if choice.message and choice.message.content:
                            content_preview = choice.message.content[:200]
                            if len(choice.message.content) > 200:
                                content_preview += "..."
                            self.debug(f"[Chat响应] 响应内容预览: {content_preview}")
                
                break
            except openai.RateLimitError as e:
                self.warning(f"[Chat请求] 第 {num_attempt + 1} 次尝试失败，原因：RateLimitError - {str(e)}")
                wait_time = parse_wait_time_from_error(e)
                if wait_time:
                    self.info(f"[Chat请求] 解析到等待时间: {wait_time}秒")
                self._wait(num_attempt, wait_time=wait_time)
                self.warning("  正在重试...")
                num_attempt += 1
            except openai.BadRequestError as e:
                self.warning(f"[Chat请求] 请求失败，原因：BadRequestError - {str(e)}")
                self.warning("  跳过此请求...")
                break
            except Exception as e:
                self.warning(f"[Chat请求] 第 {num_attempt + 1} 次尝试失败，原因：异常 - {str(e)}")
                num_attempt += 1
                if num_attempt < self._max_attempt:
                    self._wait(num_attempt)
                    self.warning("  正在重试...")
                else:
                    self.error(f"[Chat请求] 达到最大重试次数 ({self._max_attempt})，请求失败")

        # 记录总耗时
        total_duration = time.time() - request_start_time
        if response:
            self.info(f"[Chat请求] 请求完成，总耗时: {total_duration:.2f}秒，尝试次数: {num_attempt + 1}")
        else:
            self.error(f"[Chat请求] 请求最终失败，总耗时: {total_duration:.2f}秒，尝试次数: {num_attempt}")

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
                logger: 日志记录器实例
                timeout: 请求超时时间（秒）
        """
        client_config = kwargs.get("client_config", {})
        
        # 配置客户端参数
        client_configs = {
            "api_key": client_config.get("api_key") or os.environ.get("COMPANY_OPENAI_API_KEY"),
            "base_url": client_config.get("base_url") or os.environ.get("COMPANY_OPENAI_BASE_URL", "https://aigc.sankuai.com/v1/openai/native"),
            "timeout": client_config.get("timeout", kwargs.get("timeout", 120.0))  # 默认120秒超时
        }
        
        # 验证必需参数
        if not client_configs["api_key"]:
            raise ValueError("API key is required. Please set 'api_key' in client_config or 'COMPANY_OPENAI_API_KEY' environment variable.")
        
        self._client = OpenAI(**client_configs)
        self._model = kwargs.get("model", "text-embedding-ada-002")
        self._timeout = client_configs["timeout"]

        # 设置日志记录器
        self._logger = kwargs.get("logger")
        if self._logger:
            self._logger.info(f"[Embedding客户端] 初始化成功，模型: {self._model}，超时: {self._timeout}秒")
            self._logger.debug(f"[Embedding客户端] Base URL: {client_configs['base_url']}")

        # 设置缓存
        cache_config = kwargs.get("cache_config", {})
        cache_location = cache_config.get("location", None)
        if cache_location is not None:
            self._cache: PickleDB = PickleDB(location=cache_location)
            if self._logger:
                self._logger.debug(f"[Embedding客户端] 启用缓存，位置: {cache_location}")
        else:
            self._cache = None
            if self._logger:
                self._logger.debug("[Embedding客户端] 未启用缓存")
            
        # 存储可选的trace_id
        self._trace_id = client_config.get("trace_id") or os.environ.get("COMPANY_OPENAI_TRACE_ID")
        if self._trace_id and self._logger:
            self._logger.debug(f"[Embedding客户端] Trace ID: {self._trace_id}")

    def _log_info(self, message: str) -> None:
        """记录info级别日志"""
        if self._logger:
            self._logger.info(message)
        else:
            print(f"[INFO] {message}")

    def _log_debug(self, message: str) -> None:
        """记录debug级别日志"""
        if self._logger:
            self._logger.debug(message)
        else:
            print(f"[DEBUG] {message}")

    def _log_warning(self, message: str) -> None:
        """记录warning级别日志"""
        if self._logger:
            self._logger.warning(message)
        else:
            print(f"[WARNING] {message}")

    def _log_error(self, message: str) -> None:
        """记录error级别日志"""
        if self._logger:
            self._logger.error(message)
        else:
            print(f"[ERROR] {message}")

    def _save_cache(self, query: str, embedding: List[float]) -> None:
        """将embedding保存到缓存。"""
        if self._cache is None:
            return
        try:
            self._cache.set(query, embedding)
            self._log_debug(f"[Embedding缓存] 已保存，查询长度: {len(query)}, 向量维度: {len(embedding)}")
        except Exception as e:
            self._log_warning(f"[Embedding缓存] 保存失败: {str(e)}")

    def _get_cache(self, query: str) -> Union[List[float], Literal[False]]:
        """从缓存中检索embedding。"""
        if self._cache is None:
            return False
        try:
            result = self._cache.get(query)
            if result is not False:
                self._log_debug(f"[Embedding缓存] 命中缓存，查询长度: {len(query)}, 向量维度: {len(result) if result else 0}")
            return result
        except Exception as e:
            self._log_warning(f"[Embedding缓存] 读取失败: {str(e)}")
            return False

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
        # 记录请求开始时间
        request_start_time = time.time()
        
        # 确定输入类型和数量
        if isinstance(texts, str):
            input_type = "单个文本"
            input_count = 1
            text_preview = texts[:100] + "..." if len(texts) > 100 else texts
        else:
            input_type = "文本列表"
            input_count = len(texts)
            # 显示第一个文本的预览
            first_text = texts[0] if texts else ""
            text_preview = first_text[:100] + "..." if len(first_text) > 100 else first_text
        
        self._log_info(f"[Embedding请求] 开始发送embedding请求")
        self._log_debug(f"[Embedding请求] 模型: {self._model}")
        self._log_debug(f"[Embedding请求] 输入类型: {input_type}")
        self._log_debug(f"[Embedding请求] 输入数量: {input_count}")
        self._log_debug(f"[Embedding请求] 文本预览: {text_preview}")
        self._log_debug(f"[Embedding请求] 超时设置: {self._timeout}秒")
        if self._trace_id:
            self._log_debug(f"[Embedding请求] Trace ID: {self._trace_id}")
        
        retry_count = 0
        max_retries = 3
        response = None
        
        while retry_count <= max_retries:
            try:
                if retry_count > 0:
                    self._log_info(f"[Embedding请求] 第 {retry_count + 1} 次尝试")
                
                # 检查是否超过总体超时时间（防止无限重试）
                elapsed_time = time.time() - request_start_time
                if elapsed_time > self._timeout * 2:  # 总体超时是单次超时的2倍
                    self._log_error(f"[Embedding请求] 总体超时 ({elapsed_time:.1f}s > {self._timeout * 2}s)，停止重试")
                    break
                
                # 准备请求参数
                request_params = {
                    "input": texts,
                    "model": self._model,
                    "timeout": self._timeout  # 设置单次请求超时
                }
                
                # 添加额外头信息
                extra_headers = self._get_extra_headers()
                if extra_headers:
                    request_params["extra_headers"] = extra_headers
                
                # 记录单次请求开始时间
                single_request_start = time.time()
                self._log_debug(f"[Embedding请求] 发起API调用 (尝试 {retry_count + 1}/{max_retries + 1})")
                
                response = self._client.embeddings.create(**request_params)
                
                # 记录成功响应信息
                single_request_duration = time.time() - single_request_start
                self._log_info(f"[Embedding请求] 请求成功完成，单次耗时: {single_request_duration:.2f}秒")
                
                if response:
                    self._log_debug(f"[Embedding响应] 响应模型: {response.model}")
                    self._log_debug(f"[Embedding响应] 数据项数量: {len(response.data)}")
                    
                    if response.usage:
                        self._log_info(f"[Embedding响应] Token使用情况 - 输入: {response.usage.prompt_tokens}, "
                                      f"总计: {response.usage.total_tokens}")
                    
                    # 记录embedding维度信息
                    if response.data and len(response.data) > 0:
                        embedding_dim = len(response.data[0].embedding)
                        self._log_debug(f"[Embedding响应] Embedding维度: {embedding_dim}")
                
                total_duration = time.time() - request_start_time
                self._log_info(f"[Embedding请求] 请求完成，总耗时: {total_duration:.2f}秒，尝试次数: {retry_count + 1}")
                
                break
                
            except openai.RateLimitError as e:
                retry_count += 1
                elapsed_time = time.time() - single_request_start if 'single_request_start' in locals() else 0
                self._log_warning(f"[Embedding请求] 第 {retry_count} 次尝试失败，原因：RateLimitError - {str(e)}，耗时: {elapsed_time:.2f}秒")
                
                if retry_count > max_retries:
                    self._log_error(f"[Embedding请求] 达到最大重试次数 ({max_retries + 1})，停止重试")
                    break
                
                expected_wait = parse_wait_time_from_error(e)
                if expected_wait is not None:
                    self._log_info(f"[Embedding请求] 解析到等待时间: {expected_wait}秒")
                    time.sleep(expected_wait)
                else:
                    wait_time = 30
                    self._log_info(f"[Embedding请求] 使用默认等待时间: {wait_time}秒")
                    time.sleep(wait_time)
                    
                self._log_info("[Embedding请求] 准备重试...")
                    
            except openai.Timeout as e:
                retry_count += 1
                elapsed_time = time.time() - single_request_start if 'single_request_start' in locals() else 0
                self._log_warning(f"[Embedding请求] 第 {retry_count} 次尝试失败，原因：超时 - {str(e)}，耗时: {elapsed_time:.2f}秒")
                
                if retry_count > max_retries:
                    self._log_error(f"[Embedding请求] 达到最大重试次数 ({max_retries + 1})，停止重试")
                    break
                    
                wait_time = min(5 * retry_count, 30)  # 超时错误的等待时间较短
                self._log_info(f"[Embedding请求] 等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
                
            except (openai.APIConnectionError, openai.APITimeoutError) as e:
                retry_count += 1
                elapsed_time = time.time() - single_request_start if 'single_request_start' in locals() else 0
                self._log_warning(f"[Embedding请求] 第 {retry_count} 次尝试失败，原因：连接错误 - {str(e)}，耗时: {elapsed_time:.2f}秒")
                
                if retry_count > max_retries:
                    self._log_error(f"[Embedding请求] 达到最大重试次数 ({max_retries + 1})，停止重试")
                    break
                    
                wait_time = min(3 * retry_count, 20)  # 连接错误的等待时间
                self._log_info(f"[Embedding请求] 等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
                
            except KeyboardInterrupt:
                self._log_warning("[Embedding请求] 用户中断请求")
                return None
                
            except Exception as e:
                retry_count += 1
                elapsed_time = time.time() - single_request_start if 'single_request_start' in locals() else 0
                self._log_error(f"[Embedding请求] 第 {retry_count} 次尝试失败，原因：异常 - {str(e)}，耗时: {elapsed_time:.2f}秒")
                
                if retry_count > max_retries:
                    self._log_error(f"[Embedding请求] 达到最大重试次数 ({max_retries + 1})，请求失败")
                    total_duration = time.time() - request_start_time
                    self._log_error(f"[Embedding请求] 请求最终失败，总耗时: {total_duration:.2f}秒")
                    break
                    
                wait_time = min(2 ** retry_count, 60)  # 指数退避，最大60秒
                self._log_info(f"[Embedding请求] 等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
        
        # 如果所有重试都失败了，记录最终失败信息
        if response is None:
            total_duration = time.time() - request_start_time
            self._log_error(f"[Embedding请求] 所有重试失败，总耗时: {total_duration:.2f}秒，尝试次数: {retry_count}")

        return response

    def embed_documents(self, texts: List[str], batch_call: bool = False) -> List[List[float]]:
        """嵌入文档列表。"""
        start_time = time.time()
        self._log_info(f"[Embedding文档] 开始处理 {len(texts)} 个文档")
        self._log_debug(f"[Embedding文档] 批量处理模式: {'是' if batch_call else '否'}")
        
        try:
            if batch_call:
                self._log_debug("[Embedding文档] 使用批量API调用模式")
                response = self._get_response(texts)
                if response is None:
                    self._log_error("[Embedding文档] 批量请求失败，返回空向量")
                    return [[]] * len(texts)
                embeddings = [res.embedding for res in response.data]
                self._log_info(f"[Embedding文档] 批量处理完成，成功生成 {len(embeddings)} 个向量")
            else:
                self._log_debug("[Embedding文档] 使用逐个处理模式")
                embeddings = []
                successful_count = 0
                for i, text in enumerate(texts):
                    self._log_debug(f"[Embedding文档] 处理第 {i+1}/{len(texts)} 个文档")
                    try:
                        embedding = self.embed_query(text)
                        if embedding:  # 检查是否为空列表
                            successful_count += 1
                        embeddings.append(embedding)
                    except KeyboardInterrupt:
                        self._log_warning(f"[Embedding文档] 用户中断，已处理 {i}/{len(texts)} 个文档")
                        # 对剩余文档返回空向量
                        embeddings.extend([[]] * (len(texts) - i - 1))
                        break
                self._log_info(f"[Embedding文档] 逐个处理完成，成功生成 {successful_count}/{len(texts)} 个向量")
            
            duration = time.time() - start_time
            self._log_info(f"[Embedding文档] 文档嵌入完成，总耗时: {duration:.2f}秒")
            return embeddings
            
        except KeyboardInterrupt:
            self._log_warning("[Embedding文档] 用户中断文档嵌入过程")
            return [[]] * len(texts)
        except Exception as e:
            self._log_error(f"[Embedding文档] 文档嵌入过程异常: {str(e)}")
            return [[]] * len(texts)

    def embed_query(self, text: str) -> List[float]:
        """嵌入单个查询。"""
        start_time = time.time()
        text_preview = text[:50] + "..." if len(text) > 50 else text
        self._log_debug(f"[Embedding查询] 处理查询: {text_preview}")
        
        try:
            # 检查缓存
            embedding = self._get_cache(text)
            if embedding is not False:
                self._log_debug("[Embedding查询] 命中缓存，直接返回")
                duration = time.time() - start_time
                self._log_debug(f"[Embedding查询] 缓存查询完成，耗时: {duration:.3f}秒")
                return embedding
            
            self._log_debug("[Embedding查询] 缓存未命中，发起API请求")
            response = self._get_response(text)
            if response is None:
                self._log_error("[Embedding查询] API请求失败，返回空向量")
                return []
            
            embedding = response.data[0].embedding
            
            # 保存到缓存
            self._save_cache(text, embedding)
            self._log_debug("[Embedding查询] 结果已保存到缓存")
            
            duration = time.time() - start_time
            self._log_info(f"[Embedding查询] 查询嵌入完成，向量维度: {len(embedding)}，总耗时: {duration:.2f}秒")
            return embedding
            
        except KeyboardInterrupt:
            self._log_warning("[Embedding查询] 用户中断查询嵌入")
            return []
        except Exception as e:
            self._log_error(f"[Embedding查询] 查询嵌入异常: {str(e)}")
            return []