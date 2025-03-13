# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import importlib
from typing import Optional

import jsonlines
from tqdm import tqdm


def get_dataset_utils_module(dataset: str):
    module = importlib.import_module(f"open_benchmarks.dataset_utils.{dataset}")
    return module


def reformat_dataset(dataset: str, split: str, dump_path: str, dataset_dir: str, cut_off: Optional[int]=None) -> None:
    dataset_utils = get_dataset_utils_module(dataset)

    raw_data = dataset_utils.load_raw_data(dataset_dir, split)
    if cut_off is None:
        cut_off = len(raw_data)

    with jsonlines.open(dump_path, "w") as writer:
        valid_count: int = 0
        for sample in tqdm(raw_data, total=len(raw_data), desc=f'Processing {dataset}/{split}'):
            formatted_data = dataset_utils.format_raw_data(sample)
            if formatted_data is None:
                continue
            writer.write(formatted_data)
            valid_count += 1
            if valid_count >= cut_off:
                break

    print(f"Convert {valid_count} QA data from {dataset}/{split} ({len(raw_data)} originally)")
    return
