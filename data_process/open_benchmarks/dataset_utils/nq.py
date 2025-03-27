# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import unicodedata
from typing import List, Optional

import uuid
from bs4 import BeautifulSoup
from datasets import Dataset, load_dataset

from open_benchmarks.utils.question_type import infer_nq_question_type


def clean_text(text: str) -> str:
    normalized_text = unicodedata.normalize('NFKD', text)
    ascii_text = normalized_text.encode('ascii', 'ignore')
    cleaned_text = ascii_text.decode('utf-8')

    return cleaned_text


def get_answer_labels(html_bytes: bytes, short_answers: List[dict]) -> List[str]:
    answer_labels: List[str] = []

    for answer in short_answers:
        if len(answer["start_byte"]) != 0 and len(answer["end_byte"]) != 0:
            start, end = int(answer["start_byte"][0]), int(answer["end_byte"][0])
            if start > 0 and end > 0 and start < end:
                evidence: str = html_bytes[start:end].decode()
                soup = BeautifulSoup(evidence, "html.parser")
                evidence = clean_text(soup.get_text())
                answer_labels.append(evidence)

    return answer_labels


def get_evidence_contents(html_bytes: bytes, long_answer: List[dict]) -> str:
    contents = ""

    start, end = int(long_answer[0]["start_byte"]), int(long_answer[0]["end_byte"])
    if start > 0 and end > 0 and start < end:
        evidence: str = html_bytes[start:end].decode()
        soup = BeautifulSoup(evidence, "html.parser")
        evidence = clean_text(soup.get_text())
        contents = evidence

    return contents


def load_raw_data(dataset_dir: str, split: str) -> Dataset:
    dataset: Dataset = load_dataset("google-research-datasets/natural_questions", "default", split=split)
    return dataset


def format_raw_data(raw: dict) -> Optional[dict]:
    # Step 1: parse answer labels and supporting facts, validate this record.
    html_source: str = raw["document"]["html"]
    html_bytes: bytes = html_source.encode()

    answer_labels: List[str] = get_answer_labels(html_bytes, raw["annotations"]["short_answers"])
    if len(answer_labels) == 0:
        return None

    evidence_contents: str = get_evidence_contents(html_bytes, raw["annotations"]["long_answer"])
    if len(evidence_contents) == 0:
        return None

    # Step 2: re-format to fit dataset protocol.
    qtype: str = infer_nq_question_type(answer_labels, raw["annotations"]["yes_no_answer"])

    formatted_data = {
        "id": uuid.uuid4().hex,
        "question": raw["question"]["text"],
        "answer_labels": answer_labels,
        "question_type": qtype,
        "metadata": {
            "original_id": raw["id"],
            "supporting_facts": [
                {
                    "type": "wikipedia",
                    "title": raw["document"]["title"],
                    "contents": evidence_contents,
                }
            ],
            "original_type": qtype,
        },
    }

    return formatted_data
