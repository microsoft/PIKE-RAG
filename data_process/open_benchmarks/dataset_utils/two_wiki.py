# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import os
import subprocess
import zipfile
from typing import Dict, List, Optional

import uuid
import jsonlines

from open_benchmarks.dataset_utils.hotpotqa import get_supporting_facts
from open_benchmarks.utils.question_type import infer_question_type


default_name: str = "data_ids_april7.zip?rlkey=u868q6h0jojw4djjg7ea65j46"
dropbox_url: str = f"https://www.dropbox.com/scl/fi/32t7pv1dyf3o2pp0dl25u/{default_name}&e=1"


def download_raw_data(raw_dir: str) -> None:
    command = f'wget -q -P {raw_dir} {dropbox_url}'
    subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    os.rename(f"{raw_dir}/{default_name}", f"{raw_dir}/2wiki.zip")
    with zipfile.ZipFile(f"{raw_dir}/2wiki.zip", "r") as zip_ref:
        zip_ref.extractall(raw_dir)
    return


def load_raw_data(dataset_dir: str, split: str) -> List[dict]:
    raw_dir = os.path.join(dataset_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    raw_filepath = os.path.join(raw_dir, f"{split}.json")
    if not os.path.exists(raw_filepath):
        download_raw_data(raw_dir)

    with open(raw_filepath, "r") as fin:
        dataset = json.load(fin)

    return dataset


def load_title2qid(dataset_dir: str, split: str=None) -> Dict[str, str]:
    raw_dir = os.path.join(dataset_dir, "raw")

    title2qid: Dict[str, str] = {}
    with jsonlines.open(f"{raw_dir}/id_aliases.json", "r") as fin:
        for line in fin:
            qid, aliases = line["Q_id"], line["aliases"]
            for alias in aliases:
                title2qid[alias] = qid

    return title2qid


def format_raw_data(raw: dict) -> Optional[dict]:
    # Step 1: extract supporting facts contents from retrieval contexts.
    supporting_facts = get_supporting_facts(raw["supporting_facts"], raw["context"])
    if supporting_facts is None:
        return None

    # Step 2: re-format to fit dataset protocol.
    answer_labels: List[str] = [raw["answer"]]
    qtype: str = infer_question_type(answer_labels)

    formatted_data = {
        "id": uuid.uuid4().hex,
        "question": raw["question"],
        "answer_labels": answer_labels,
        "question_type": qtype,
        "metadata": {
            "original_id": raw["_id"],
            "original_type": raw["type"],
            "supporting_facts": supporting_facts,
            "retrieval_contexts": [
                {
                    "type": "wikipedia",
                    "title": title,
                    "contents": "".join(sentences),
                }
                for title, sentences in raw["context"]
            ],
            "reasoning_logics": [
                {
                    "type": "wikidata",
                    "title": title,
                    "section": section,
                    "contents": content,
                }
                for title, section, content in raw["evidences"]
            ],
        }
    }

    return formatted_data
