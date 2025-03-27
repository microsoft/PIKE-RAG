# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import os
from functools import partial
from typing import Dict, List, Optional

import yaml

from open_benchmarks.reformat_dataset import reformat_dataset
from open_benchmarks.sample_dataset import sample_datasets
from open_benchmarks.utils.filepaths import get_dataset_dir, get_document_dir, get_split_filepath
from open_benchmarks.utils.stats import check_dataset_split


def load_yaml_config(config_path: str, args: argparse.Namespace) -> dict:
    with open(config_path, "r") as fin:
        yaml_config: dict = yaml.safe_load(fin)

    return yaml_config


def create_dirs(root_dir: str, datasets: List[str]) -> None:
    # Create saving directories if not exist.
    if not os.path.exists(root_dir):
        print(f"Create directory for dataset saving: {root_dir}")
        os.makedirs(root_dir, exist_ok=True)

    for dataset in datasets:
        dataset_dir = get_dataset_dir(root_dir, dataset)
        os.makedirs(dataset_dir, exist_ok=True)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="the path of the yaml config file you want to use")
    args = parser.parse_args()

    yaml_config: dict = load_yaml_config(args.config, args)

    # Read yaml configs
    root_save_dir: str = yaml_config["root_save_dir"]
    running_modes: Dict[str, bool] = yaml_config["running_modes"]
    dataset2split: Dict[str, str] = yaml_config["datasets"]

    # Check dataset split setting.
    for dataset, split in dataset2split.items():
        check_dataset_split(dataset, split)

    # Create directories for processed data saving.
    create_dirs(root_save_dir, list(dataset2split.keys()))

    # Build up QA data.
    if running_modes["build_split"]:
        cut_off: Optional[int] = yaml_config["cut_off"]
        for dataset, split in dataset2split.items():
            dataset_dir: str = get_dataset_dir(root_save_dir, dataset)
            split_path: str = get_split_filepath(root_save_dir, dataset, split, sample_num=None)
            reformat_dataset(dataset, split, split_path, dataset_dir, cut_off)

    # Sample and download valid samples and docs for each dataset.
    if running_modes["sample_sets"]:
        # Get and check the random seed.
        random_seed: int = yaml_config["seed"]
        assert isinstance(random_seed, int), (
            f"Valid int must be provided as `seed` for random sampling but get {random_seed}"
        )

        # Initialize sample size list, from small to large.
        sample_size_list: List[int] = list(range(100, 1001, 100)) + list(range(2000, 150001, 1000))

        # Get the unified document dir.
        unified_doc_dir: str = get_document_dir(root_save_dir)

        # Sample by dataset one by one
        for dataset, split in dataset2split.items():
            sample_datasets(
                dataset, split,
                sample_size_list=sample_size_list,
                random_seed=random_seed,
                document_dir=unified_doc_dir,
                split_path_func=partial(get_split_filepath, root_dir=root_save_dir, dataset=dataset, split=split),
            )
