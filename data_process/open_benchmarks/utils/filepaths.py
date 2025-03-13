# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from typing import Dict, Literal, Optional

from open_benchmarks.utils.stats import FILE_TYPES_TO_DOWNLOAD, SOURCE_TYPES_TO_DOWNLOAD


def get_dataset_dir(root_dir: str, dataset: str) -> str:
    return os.path.join(root_dir, dataset)


def get_split_filepath(root_dir: str, dataset: str, split: str, sample_num: Optional[int]) -> str:
    if sample_num is None:
        filepath = os.path.join(root_dir, dataset, f"{split}.jsonl")
    else:
        filepath = os.path.join(root_dir, dataset, f"{split}_{sample_num}.jsonl")
    return filepath


def get_document_dir(root_dir: str) -> str:
    doc_dir = os.path.join(root_dir, "documents")

    # Create dirs for each source type and file type.
    for source_type in SOURCE_TYPES_TO_DOWNLOAD:
        for file_type in FILE_TYPES_TO_DOWNLOAD:
            dir = os.path.join(doc_dir, source_type, file_type)
            if not os.path.exists(dir):
                os.makedirs(dir)

    return doc_dir


def get_doc_location_filepath(root_dir: str) -> str:
    filepath = os.path.join(root_dir, "doc_title_type_to_location.json")
    return filepath


def get_title_status_filepath(root_dir: str) -> str:
    filepath = os.path.join(root_dir, "wiki_title_type_to_validation_status.json")
    return filepath


def title_to_filename_prefix(title: str) -> str:
    return title.replace("/", " ")


def get_download_filepaths(title: str, source_type: str, document_dir: str) -> Dict[Literal["pdf", "html"], str]:
    filename_prefix = title_to_filename_prefix(title)
    filepaths = {
        filetype: os.path.join(document_dir, source_type, filetype, f"{filename_prefix}.{filetype}")
        for filetype in FILE_TYPES_TO_DOWNLOAD
    }
    return filepaths
