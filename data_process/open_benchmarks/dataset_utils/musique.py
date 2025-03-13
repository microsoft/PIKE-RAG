# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
import os
import subprocess
from typing import Dict, List, Literal, Optional

import uuid
import jsonlines

from open_benchmarks.utils.question_type import infer_question_type


zipfile_id = "1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h"


def download_raw_data(raw_dir: str) -> None:
    command: str = "pip3 install -q gdown"
    subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    output_path = f"{raw_dir}/musique.zip"
    command = f"gdown --id {zipfile_id} --output {output_path} && unzip {output_path} -d {raw_dir}"
    subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    command = f"mv {raw_dir}/data/* {raw_dir}"
    subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return


def load_raw_data(dataset_dir: str, split: str) -> List[dict]:
    raw_dir = os.path.join(dataset_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    raw_filepath = os.path.join(raw_dir, f"musique_ans_v1.0_{split}.jsonl")
    if not os.path.exists(raw_filepath):
        download_raw_data(raw_dir)

    with jsonlines.open(raw_filepath, "r") as reader:
        dataset = [data for data in reader]

    return dataset


def format_raw_data(raw: dict) -> Optional[dict]:
    # Step 1: Extract contents of Retrieved Contexts and Supporting Facts
    retrieval_contexts: List[Dict[Literal["type", "title", "contents"], str]] = [
        {
            "type": "wikipedia",
            "title": paragraph["title"],
            "contents": paragraph["paragraph_text"],
        }
        for paragraph in raw["paragraphs"]
    ]

    supporting_facts: List[Dict[Literal["type", "title", "contents"], str]] = [
        copy.deepcopy(retrieval_contexts[item["paragraph_support_idx"]])
        for item in raw["question_decomposition"]
    ]

    # Step 3: convert to data protocol
    answer_labels: List[str] = [raw["answer"]] + raw["answer_aliases"]
    qtype: str = infer_question_type(answer_labels)

    formatted_data = {
        "id": uuid.uuid4().hex,
        "question": raw["question"],
        "answer_labels": answer_labels,
        "question_type": qtype,
        "metadata": {
            "original_id": raw["id"],
            "supporting_facts": supporting_facts,
            "retrieval_contexts": retrieval_contexts,
        }
    }

    return formatted_data
