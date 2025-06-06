# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List

import jsonlines
from dacite import from_dict
from tqdm import tqdm

from pikerag.utils.logger import Logger
from pikerag.workflows.common import GenerationQaData
from pikerag.workflows.evaluation.evaluator import Evaluator
from pikerag.utils.monitor import create_monitor
import os


class EvaluationWorkflow:
    def __init__(self, yaml_config: dict) -> None:
        self._yaml_config: dict = yaml_config

        self._init_logger()

        self._init_monitor()

        self._load_result_jsonlines()

        self._init_evaluator()

    def _init_logger(self) -> None:
        self._logger: Logger = Logger(name="evaluation", dump_folder=self._yaml_config["log_dir"])

    def _init_monitor(self) -> None:
        """初始化API监控器"""
        monitoring_config = self._yaml_config.get("monitoring", {})
        enable_monitoring = monitoring_config.get("enable", True)
        
        if enable_monitoring:
            self._monitor = create_monitor(
                logger=self._logger,
                window_minutes=monitoring_config.get("rpm_window_minutes", 1)
            )
            if hasattr(self, '_logger') and self._logger:
                self._logger.info(f"✅ API监控器已在 {self.__class__.__name__} 中启用", tag=self.__class__.__name__)
        else:
            self._monitor = None
            if hasattr(self, '_logger') and self._logger:
                self._logger.info(f"❌ API监控器已在 {self.__class__.__name__} 中禁用", tag=self.__class__.__name__)

    def _load_result_jsonlines(self) -> None:
        filepath = self._yaml_config["result_path"]
        with jsonlines.open(filepath, "r") as reader:
            self._results: List[GenerationQaData] = [
                from_dict(GenerationQaData, data)
                for data in reader
            ]

    def _init_evaluator(self):
        evaluator_config: dict = self._yaml_config.get("evaluator", {})

        self._evaluator = Evaluator(
            evaluator_config=evaluator_config,
            num_rounds=self._yaml_config["test_rounds"],
            num_data=self._yaml_config["num_test_data"],
            log_dir=self._yaml_config["log_dir"],
            main_logger=self._logger,
        )
        return

    def run(self) -> None:
        """Main entry of the evaluation workflow."""
        if hasattr(self, '_monitor') and self._monitor:
            self._monitor.start_monitoring()
            if hasattr(self, '_logger') and self._logger:
                self._logger.info(f"🚀 开始 {self.__class__.__name__} 任务", tag=self.__class__.__name__)

        try:
            self._evaluator.evaluate(self._qa_dataset, self._main_metric_name)
            self._logger.info("Evaluation finished.")

        except Exception as e:
            if hasattr(self, '_logger') and self._logger:
                self._logger.error(f"{self.__class__.__name__} 任务执行失败: {e}", tag=self.__class__.__name__)
            raise
        finally:
            if hasattr(self, '_monitor') and self._monitor:
                self._monitor.stop_monitoring()
                # Note: The summary will reflect any LLM calls made by metrics during evaluation.
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
