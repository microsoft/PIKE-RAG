# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
from typing import Dict, List, Optional

import uuid
from datasets import Dataset, load_dataset
from tqdm import tqdm

from open_benchmarks.utils.question_type import infer_question_type


def load_raw_data(dataset_dir: str, split: str) -> Dataset:
    dataset: Dataset = load_dataset("akariasai/PopQA", split=split)
    return dataset


def format_raw_data(raw: dict) -> Optional[dict]:
    # Skip raw if subject item not exist.
    if raw["subj"] is None:
        return None

    # Re-format to fit dataset protocol.
    answer_labels: List[str] = json.loads(raw["possible_answers"])
    qtype: str = infer_question_type(answer_labels)

    formatted_data = {
        "id": uuid.uuid4().hex,
        "question": raw["question"],
        "answer_labels": answer_labels,
        "question_type": qtype,
        "metadata": {
            "original_id": raw["id"],
            "supporting_facts": [
                {
                    "type": "wikidata",
                    "title": raw["subj"],
                    "section": raw["prop"],
                    "contents": raw["obj"],
                }
            ],
        }
    }

    return formatted_data


def extract_title2qid(split: str, dump_path: str) -> None:
    raw_data = load_raw_data("", split)

    wikidata_title2qid: Dict[str, str] = {}
    for raw in tqdm(raw_data, total=len(raw_data), desc=f"Processing PopQA/{split}"):
        title = raw["subj"]
        qid = raw["s_uri"].split("/")[-1]
        wikidata_title2qid[title] = qid

    with open(dump_path, "w", encoding="utf-8") as fout:
        json.dump(wikidata_title2qid, fout, ensure_ascii=False)

    return


def load_title2qid(dataset_dir: str, split: str) -> Dict[str, str]:
    dataset = load_raw_data("", split)

    title2qid = {}
    for raw in dataset:
        if raw["subj"] is not None:
            title = raw["subj"]
            qid = raw["s_uri"].split("/")[-1]
            title2qid[title] = qid

    return title2qid
