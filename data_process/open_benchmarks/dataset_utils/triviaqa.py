# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Optional

import uuid
from datasets import Dataset, load_dataset

from open_benchmarks.utils.question_type import infer_question_type


def load_raw_data(dataset_dir: str, split: str) -> Dataset:
    dataset: Dataset = load_dataset("mandarjoshi/trivia_qa", "rc", split=split)
    return dataset


def format_raw_data(raw: dict) -> Optional[dict]:
    # Step 1: extract contents of BingSearch
    bing_search_results = []
    for title, url, description, contents, rank in zip(
        raw["search_results"]["title"],
        raw["search_results"]["url"],
        raw["search_results"]["description"],
        raw["search_results"]["search_context"],
        raw["search_results"]["rank"],
    ):
        bing_search_results.append(
            {
                "type": "BingSearch",
                "title": title,
                "url": url,
                "description": description,
                "contents": contents,
                "rank": rank,
            }
        )

    # Step 2: re-format to fit dataset protocol.
    answer_labels: List[str] = raw["answer"]["aliases"]
    qtype: str = infer_question_type(answer_labels)

    formatted_data = {
        "id": uuid.uuid4().hex,
        "question": raw["question"],
        "answer_labels": answer_labels,
        "question_type": qtype,
        "metadata": {
            "original_id": raw["question_id"],
            "retrieval_contexts": [
                {
                    "type": "wikipedia",
                    "title": title,
                }
                for title in raw["entity_pages"]["title"]
            ] + bing_search_results,
        },
    }

    return formatted_data
