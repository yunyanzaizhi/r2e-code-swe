import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs


def build_rows(size: int, split: str):
    rows = []
    for idx in range(size):
        rows.append(
            {
                "data_source": "r2e_code_swe",
                "prompt": [{"role": "user", "content": ""}],
                "ability": "agent",
                "extra_info": {"split": split, "index": idx},
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Create lightweight parquet placeholders for R2E code/SWE env rollouts.")
    parser.add_argument("--local_dir", default="~/data/verl-agent/r2e_code_swe")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--train_data_size", default=20, type=int)
    parser.add_argument("--val_data_size", default=3, type=int)
    args = parser.parse_args()

    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    train_dataset = datasets.Dataset.from_list(build_rows(args.train_data_size, "train"))
    val_dataset = datasets.Dataset.from_list(build_rows(args.val_data_size, "dev_10pr_v1"))

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    val_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)

    print(f"Wrote R2E code/SWE placeholder data to {local_dir}")


if __name__ == "__main__":
    main()
