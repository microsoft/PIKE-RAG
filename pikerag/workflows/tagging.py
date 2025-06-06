# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import importlib
import os
from typing import List, Tuple

from tqdm import tqdm

from pikerag.document_transformers import LLMPoweredTagger
from pikerag.llm_client import BaseLLMClient
from pikerag.utils.config_loader import load_protocol
from pikerag.utils.logger import Logger
from pikerag.utils.walker import list_files_recursively
from pikerag.utils.monitor import create_monitor, get_global_monitor


class TaggingWorkflow:
    def __init__(self, yaml_config: dict) -> None:
        self._yaml_config: dict = yaml_config

        self._init_logger()

        # 初始化监控器
        self._init_monitor()

        self._init_tagger()

        self._init_file_infos()
        self._init_file_loader_and_saver()
        return

    def _init_logger(self) -> None:
        self._logger: Logger = Logger(
            name=self._yaml_config["experiment_name"],
            dump_folder=self._yaml_config["log_dir"],
        )

    def _init_monitor(self) -> None:
        """初始化API监控器"""
        # 创建监控器
        monitoring_config = self._yaml_config.get("monitoring", {})
        enable_monitoring = monitoring_config.get("enable", True)
        
        if enable_monitoring:
            self._monitor = create_monitor(
                logger=self._logger,
                window_minutes=monitoring_config.get("rpm_window_minutes", 1)
            )
            self._logger.info("✅ API监控器已启用", tag="TaggingWorkflow")
        else:
            self._monitor = None
            self._logger.info("❌ API监控器已禁用", tag="TaggingWorkflow")

    def _init_llm_client(self) -> None:
        # Dynamically import the LLM client.
        self._client_logger = Logger(name="client", dump_mode="a", dump_folder=self._yaml_config["log_dir"])

        llm_client_config = self._yaml_config["llm_client"]
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
            llm_config=llm_client_config["llm_config"],
            **llm_client_config.get("args", {}),
        )
        
        # 设置监控器到客户端
        if self._monitor:
            self._client.set_monitor(self._monitor)

        return

    def _init_tagger(self) -> None:
        self._init_llm_client()

        tagger_config: dict = self._yaml_config["tagger"]

        # Dynamically import the tagging communication protocol
        self._tagging_protocol = load_protocol(
            module_path=tagger_config["tagging_protocol"]["module_path"],
            protocol_name=tagger_config["tagging_protocol"]["attr_name"],
        )

        self._tag_name: str = tagger_config["tag_name"]
        self._tagger_logger = Logger(name="tagger", dump_mode="w", dump_folder=self._yaml_config["log_dir"])

        self._tagger = LLMPoweredTagger(
            llm_client=self._client,
            tagging_protocol=self._tagging_protocol,
            num_parallel=tagger_config.get("num_parallel", 1),
            tag_name=self._tag_name,
            llm_config=self._yaml_config["llm_client"]["llm_config"],
            logger=self._tagger_logger,
        )

        return

    def _init_file_infos(self) -> None:
        input_setting: dict = self._yaml_config.get("input_doc_setting", None)
        output_setting: dict = self._yaml_config.get("output_doc_setting", None)

        if input_setting is None or output_setting is None:
            self._file_infos = None
            return

        input_file_infos = list_files_recursively(
            directory=input_setting.get("doc_dir"),
            extensions=input_setting.get("extensions"),
        )

        output_dir = output_setting.get("doc_dir")
        output_suffix = output_setting.get("suffix")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        self._file_infos: List[Tuple[str, str, str]] = [
            (doc_name, doc_path, os.path.join(output_dir, f"{os.path.splitext(doc_name)[0]}.{output_suffix}"))
            for doc_name, doc_path in input_file_infos
        ]
        return

    def _init_file_loader_and_saver(self) -> None:
        self._load_func = getattr(
            importlib.import_module(self._yaml_config["ori_doc_loading"]["module"]),
            self._yaml_config["ori_doc_loading"]["name"],
        )
        self._load_args = self._yaml_config["ori_doc_loading"].get("args", {})
        self._save_func = getattr(
            importlib.import_module(self._yaml_config["tagged_doc_saving"]["module"]),
            self._yaml_config["tagged_doc_saving"]["name"],
        )
        self._save_args = self._yaml_config["tagged_doc_saving"].get("args", {})
        return

    def _run_multi(self) -> None:
        for doc_name, input_path, output_path in tqdm(self._file_infos, desc="Tagging file"):
            if os.path.exists(output_path) is True:
                self._logger.info(f"Skip file: {doc_name} due to output already exist!")
                continue

            self._logger.info(f"Loading file: {doc_name}")
            docs = self._load_func(input_path)
            tagged_docs = self._tagger.transform_documents(docs)
            self._save_func(tagged_docs, output_path)

    def _run_single(self) -> None:
        docs = self._load_func(**self._load_args)
        tagged_docs = self._tagger.transform_documents(docs)
        self._save_func(tagged_docs, **self._save_args)

    def run(self) -> None:
        # 开始监控
        if self._monitor:
            self._monitor.start_monitoring()
            self._logger.info("🚀 开始标注任务", tag="TaggingWorkflow")

        try:
            if self._file_infos is None:
                self._run_single()
            else:
                self._run_multi()
        except Exception as e:
            self._logger.error(f"标注任务执行失败: {e}", tag="TaggingWorkflow")
            raise
        finally:
            # 结束监控并生成报告
            if self._monitor:
                self._monitor.stop_monitoring()
                
                # 打印统计摘要
                self._monitor.print_summary()
                
                # 保存详细报告
                report_path = os.path.join(self._yaml_config["log_dir"], "api_monitoring_report.json")
                self._monitor.save_detailed_report(report_path)
                
                self._logger.info("✅ 标注任务完成", tag="TaggingWorkflow")

        return
