# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional

import uuid
from datasets import Dataset, load_dataset

from open_benchmarks.utils.question_type import infer_question_type


def load_raw_data(dataset_dir: str, split: str) -> Dataset:
    dataset: Dataset = load_dataset("Stanford/web_questions", split=split)
    return dataset


def format_raw_data(raw: dict) -> Optional[dict]:
    # Re-format to fit dataset protocol.
    qtype: str = infer_question_type(raw["answers"])

    formatted_data = {
        "id": uuid.uuid4().hex,
        "question": raw["question"],
        "answer_labels": raw["answers"],
        "question_type": qtype,
        "metadata": {
            "supporting_facts": [
                {
                    "type": "wikipedia",
                    "title": raw["url"].split("/")[-1].replace("_", " "),
                }
            ]
        },
    }
    return formatted_data
