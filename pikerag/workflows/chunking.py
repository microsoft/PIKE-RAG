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
from pikerag.utils.config_loader import load_class
from pikerag.utils.logger import Logger
from pikerag.utils.walker import list_files_recursively


class ChunkingWorkflow:
    def __init__(self, yaml_config: dict) -> None:
        self._yaml_config: dict = yaml_config

        self._init_logger()
        self._init_splitter()

        self._init_file_infos()
        return

    def _init_logger(self) -> None:
        self._logger: Logger = Logger(
            name=self._yaml_config["experiment_name"],
            dump_folder=self._yaml_config["log_dir"],
        )

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
        return

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
            self._init_llm_client()

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
        for doc_name, input_path, output_path in tqdm(self._file_infos, desc="Chunking file"):
            if os.path.exists(output_path) is True:
                self._logger.info(f"Skip file: {doc_name} due to output already exist!")
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

            # Dump document chunks to disk.
            with open(output_path, "wb") as fout:
                pickle.dump(chunk_docs, fout)
