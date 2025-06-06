# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import importlib
import os
from typing import List, Tuple

import pickle
from tqdm import tqdm

from pikerag.document_loaders import get_loader
from pikerag.document_transformers import LLMPoweredRecursiveSplitter
from pikerag.llm_client import BaseLLMClient
from pikerag.utils.config_loader import load_class, load_protocol
from pikerag.utils.logger import Logger
from pikerag.utils.walker import list_files_recursively
from pikerag.utils.monitor import create_monitor


class ChunkingWorkflow:
    def __init__(self, yaml_config: dict) -> None:
        self._yaml_config: dict = yaml_config

        self._init_logger()
        self._init_splitter()

        self._init_file_infos()
        self._init_monitor()
        self._init_llm_client_if_needed()
        return

    def _init_logger(self) -> None:
        self._logger: Logger = Logger(
            name=self._yaml_config["experiment_name"],
            dump_folder=self._yaml_config["log_dir"],
        )

    def _init_monitor(self) -> None:
        """初始化API监控器"""
        monitoring_config = self._yaml_config.get("monitoring", {})
        enable_monitoring = monitoring_config.get("enable", True)
        
        if enable_monitoring:
            self._monitor = create_monitor(
                logger=self._logger,
                window_minutes=monitoring_config.get("rpm_window_minutes", 1),
                print_interval=monitoring_config.get("print_interval", 60)
            )
            if hasattr(self, '_logger') and self._logger:
                print_interval = monitoring_config.get("print_interval", 60)
                if print_interval > 0:
                    self._logger.info(f"✅ API监控器已在 {self.__class__.__name__} 中启用（实时统计间隔: {print_interval}秒）", tag=self.__class__.__name__)
                else:
                    self._logger.info(f"✅ API监控器已在 {self.__class__.__name__} 中启用（禁用实时统计）", tag=self.__class__.__name__)
        else:
            self._monitor = None
            if hasattr(self, '_logger') and self._logger:
                self._logger.info(f"❌ API监控器已在 {self.__class__.__name__} 中禁用", tag=self.__class__.__name__)

    def _init_llm_client_if_needed(self) -> None:
        """Initializes LLM client if an LLM-powered chunker is used."""
        self._client = None
        chunker_config = self._yaml_config.get("chunker", {})
        llm_powered_chunker_config = chunker_config.get("llm_powered_chunker_config")

        if llm_powered_chunker_config:
            self._logger.info("LLM-powered chunker configured, initializing LLM client.")
            llm_client_config = self._yaml_config.get("llm_client")
            if not llm_client_config:
                raise ValueError("llm_client configuration is required for llm_powered_chunker_config")

            self._client_logger = Logger(name="client", dump_mode="a", dump_folder=self._yaml_config["log_dir"])
            cache_location = os.path.join(
                self._yaml_config["log_dir"],
                f"{llm_client_config['cache_config']['location_prefix']}.db",
            )

            client_module = importlib.import_module(llm_client_config["module_path"])
            client_class = getattr(client_module, llm_client_config["class_name"])
            assert issubclass(client_class, BaseLLMClient)
            self._client = client_class(
                location=cache_location,
                auto_dump=llm_client_config["cache_config"]["auto_dump"],
                logger=self._client_logger,
                llm_config=llm_client_config.get("llm_config", {}),
                **llm_client_config.get("args", {}),
            )
            # 设置监控器到客户端
            if hasattr(self, '_monitor') and self._monitor and hasattr(self._client, 'set_monitor'):
                self._client.set_monitor(self._monitor)
            self._logger.info("LLM Client Inited for chunker.")
        else:
            self._logger.info("No LLM-powered chunker configured, LLM client not initialized.")

    def _init_splitter(self) -> None:
        splitter_config: dict = self._yaml_config["splitter"]
        splitter_args: dict = splitter_config.get("args", {})

        splitter_class = load_class(
            module_path=splitter_config["module_path"],
            class_name=splitter_config["class_name"],
            base_class=None,
        )

        if issubclass(splitter_class, (LLMPoweredRecursiveSplitter)):
            # Initialize LLM client
            self._init_llm_client_if_needed()

            # Update args
            splitter_args["llm_client"] = self._client
            splitter_args["llm_config"] = self._yaml_config["llm_client"]["llm_config"]

            splitter_args["logger"] = self._logger

        if issubclass(splitter_class, LLMPoweredRecursiveSplitter):
            # Load protocols
            protocol_configs = self._yaml_config["chunking_protocol"]
            protocol_module = importlib.import_module(protocol_configs["module_path"])
            chunk_summary_protocol = getattr(protocol_module, protocol_configs["chunk_summary"])
            chunk_summary_refinement_protocol = getattr(protocol_module, protocol_configs["chunk_summary_refinement"])
            chunk_resplit_protocol = getattr(protocol_module, protocol_configs["chunk_resplit"])

            # Update args
            splitter_args["first_chunk_summary_protocol"] = chunk_summary_protocol
            splitter_args["last_chunk_summary_protocol"] = chunk_summary_refinement_protocol
            splitter_args["chunk_resplit_protocol"] = chunk_resplit_protocol

        self._splitter = splitter_class(**splitter_args)
        return

    def _init_file_infos(self) -> None:
        input_setting: dict = self._yaml_config.get("input_doc_setting")
        output_setting: dict = self._yaml_config.get("output_doc_setting")
        assert input_setting is not None and output_setting is not None, (
            f"input_doc_setting and output_doc_setting should be provided!"
        )

        input_file_infos = list_files_recursively(
            directory=input_setting.get("doc_dir"),
            extensions=input_setting.get("extensions"),
        )

        output_dir = output_setting.get("doc_dir")
        output_suffix = output_setting.get("suffix", "pkl")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        self._file_infos: List[Tuple[str, str, str]] = [
            (doc_name, doc_path, os.path.join(output_dir, f"{os.path.splitext(doc_name)[0]}.{output_suffix}"))
            for doc_name, doc_path in input_file_infos
        ]
        return

    def run(self) -> None:
        """Main entry of the chunking workflow. It will load, chunk, and save docs based on the config."""
        # ADDED MONITORING WRAPPER
        if hasattr(self, '_monitor') and self._monitor:
            self._monitor.start_monitoring()
            if hasattr(self, '_logger') and self._logger:
                self._logger.info(f"🚀 开始 {self.__class__.__name__} 任务", tag=self.__class__.__name__)
        
        try:
            processed_doc_paths = set()
            if os.path.exists(self._chunk_saver.get_record_file_path()):
                with open(self._chunk_saver.get_record_file_path(), "r") as fin:
                    for line in fin:
                        processed_doc_paths.add(line.strip())
            self._logger.info(f"{len(processed_doc_paths)} docs found in record file, will be skipped.")

            all_chunks = []
            for doc_name, input_path, output_path in tqdm(self._file_infos, desc="Chunking file"):
                if input_path in processed_doc_paths:
                    self._logger.info(f"Skip file {doc_name} as it's already processed.")
                    continue

                self._logger.info(f"Loading file: {doc_name}")

                # Try get the file loader and load documents
                doc_loader = get_loader(file_path=input_path, file_type=None)
                if doc_loader is None:
                    self._logger.info(f"Skip file {doc_name} due to undefined Document Loader.")
                    continue
                docs = doc_loader.load()

                # Add metadata
                for doc in docs:
                    doc.metadata.update({"filename": doc_name})

                # Document Splitting
                chunk_docs = self._splitter.transform_documents(docs)
                all_chunks.extend(chunk_docs)

                self._chunk_saver.save_record(input_path)

            self._logger.info(f"Total {len(all_chunks)} chunks generated.")
            self._chunk_saver.save_chunks(all_chunks)

        except Exception as e:
            if hasattr(self, '_logger') and self._logger:
                self._logger.error(f"{self.__class__.__name__} 任务执行失败: {e}", tag=self.__class__.__name__)
            raise
        finally:
            if hasattr(self, '_monitor') and self._monitor:
                self._monitor.stop_monitoring()
                self._monitor.print_summary()
                
                log_dir = self._yaml_config.get("log_dir", ".")
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)
                report_filename = f"{self.__class__.__name__}_api_monitoring_report.json"
                report_path = os.path.join(log_dir, report_filename)
                self._monitor.save_detailed_report(report_path)
                
                if hasattr(self, '_logger') and self._logger:
                    self._logger.info(f"✅ {self.__class__.__name__} 任务完成", tag=self.__class__.__name__)
        return
