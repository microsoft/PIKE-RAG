# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import os
import requests
from typing import Dict, List, Literal, Optional, Tuple

import uuid

from open_benchmarks.utils.question_type import infer_question_type


split2url: Dict[str, str] = {
    "train": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json",
    "dev": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
}


def download_raw_data(raw_filepath: str, split: str) -> None:
    url = split2url[split]
    with requests.get(url) as response:
        with open(raw_filepath, "wb") as fout:
            for chunk in response.iter_content(chunk_size=1024):
                fout.write(chunk)
    return


def load_raw_data(dataset_dir: str, split: str) -> List[dict]:
    raw_dir = os.path.join(dataset_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    raw_filepath = os.path.join(raw_dir, f"{split}.json")
    if not os.path.exists(raw_filepath):
        download_raw_data(raw_filepath, split)

    with open(raw_filepath, "r", encoding="utf-8") as fin:
        dataset = json.load(fin)
    return dataset


def get_idx_sentence(
    title: str, idx: int, original_contexts: List[Tuple[str, List[str]]], verbose: bool=False,
) -> Optional[str]:
    for item_title, sentences in original_contexts:
        if item_title == title and idx < len(sentences):
            return sentences[idx]

    if verbose:
        print(f"######## Indexed sentence not found ########")
        print(f"title: {title}, idx: {idx}")
        for item_title, sentences in original_contexts:
            if item_title != title:
                continue
            for i, sentence in enumerate(sentences):
                print(f"  {i}: {sentence}")
            print()
    return None


def get_supporting_facts(
    supporting_fact_tuples: List[Tuple[str, int]],
    context_tuples: List[Tuple[str, List[str]]],
) -> Optional[List[Dict[Literal["type", "title", "contents"], str]]]:
    supporting_facts: List[dict] = []
    for title, sent_idx in supporting_fact_tuples:
        content: Optional[str] = get_idx_sentence(title, sent_idx, context_tuples)
        if content is None:
            return None
        supporting_facts.append(
            {
                "type": "wikipedia",
                "title": title,
                "contents": content,
            }
        )
    return supporting_facts


def format_raw_data(raw: dict) -> Optional[dict]:
    # Step 1: extract supporting facts contents from retrieval contexts.
    # Skip sample if supporting fact not found. Currently there is one error case in `dev` split with `_id`:
    # 5ae61bfd5542992663a4f261
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
            "supporting_facts": supporting_facts,
            "retrieval_contexts": [
                {
                    "type": "wikipedia",
                    "title": title,
                    "contents": "".join(sentences),
                }
                for title, sentences in raw["context"]
            ],
            "original_type": raw["type"],
            "original_level": raw["level"],
        },
    }

    return formatted_data
