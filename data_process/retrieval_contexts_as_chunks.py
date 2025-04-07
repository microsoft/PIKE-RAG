# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from typing import Dict, List, Literal

import jsonlines
import numpy as np


def read_testing_suite(filepath: str) -> List[dict]:
    with jsonlines.open(filepath, "r") as reader:
        testing_suite = [data for data in reader]
    return testing_suite


def get_chunks_from_testing_suite(testing_suite: List[dict]) -> Dict[str, List[str]]:
    chunks_by_title = {}
    for qa in testing_suite:
        for retrieval_context in qa["metadata"]["retrieval_contexts"]:
            title = retrieval_context["title"]
            contents = retrieval_context["contents"]
            if title not in chunks_by_title:
                chunks_by_title[title] = []
            chunks_by_title[title].append(contents)
    chunks_by_title = {title: list(set(chunks)) for title, chunks in chunks_by_title.items()}

    chunk_count = [len(lst) for _, lst in chunks_by_title.items()]
    print(
        f"{len(chunks_by_title)} titles in total. "
        f"{sum(chunk_count)} chunks in total. "
        f"Chunk count: {min(chunk_count)} ~ {max(chunk_count)}, avg: {np.mean(chunk_count)}"
    )
    return chunks_by_title


if __name__ == "__main__":
    data_dir = "data"
    datasets = ["musique"]
    for dataset in datasets:
        dataset_dir = os.path.join(data_dir, dataset)
        input_path = os.path.join(dataset_dir, "dev_500.jsonl")
        output_path = os.path.join(dataset_dir, "dev_500_retrieval_contexts_as_chunks.jsonl")

        print(f"\n#### Dataset: {dataset}")
        testing_suite = read_testing_suite(input_path)
        chunks_by_title = get_chunks_from_testing_suite(testing_suite)
        chunk_dicts: List[Dict[Literal["chunk_id", "title", "content"], str]] = [
            {
                "chunk_id": f"{title}-{cidx}-{len(chunks)}",
                "title": title,
                "content": chunk,
            }
            for title, chunks in chunks_by_title.items()
            for cidx, chunk in enumerate(chunks)
        ]

        with jsonlines.open(output_path, "w") as writer:
            writer.write_all(chunk_dicts)

        counter = {}
        for qa in testing_suite:
            count = len(qa["metadata"]["retrieval_contexts"])
            counter[count] = counter.get(count, 0) + 1
        print("Retrieval Contexts:", counter)

        counter = {}
        for qa in testing_suite:
            count = len(qa["metadata"]["supporting_facts"])
            counter[count] = counter.get(count, 0) + 1
        print("Supporting Facts:", counter)

"""
#### Dataset: hotpotqa
4949 titles in total. 4950 chunks in total. Chunk count: 1 ~ 2, avg: 1.0002020610224287
Retrieval Contexts: {10: 497, 5: 1, 2: 2}
Supporting Facts: {2: 328, 4: 37, 3: 126, 5: 7, 6: 2}

#### Dataset: two_wiki
3410 titles in total. 3410 chunks in total. Chunk count: 1 ~ 1, avg: 1.0
Retrieval Contexts: {10: 500}
Supporting Facts: {4: 107, 2: 391, 5: 1, 3: 1}

#### Dataset: musique
6075 titles in total. 7120 chunks in total. Chunk count: 1 ~ 19, avg: 1.1720164609053498
Retrieval Contexts: {20: 496, 19: 3, 17: 1}
Supporting Facts: {2: 263, 4: 68, 3: 169}
"""
