from pathlib import Path
from tqdm import tqdm
import json
import datasets
from typing import Dict, Type, List
import argparse


def parse_cla():
    """parses command-line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-save_path", type=str)
    return parser.parse_args()

def load_dataset() -> Type[datasets.Dataset]:
    return datasets.load_dataset("Kuugo/chinese_law_ft_dataset",trust_remote_code=True)


# def save_jsonl(dataset:Type[datasets.Dataset], save_path:str):
#     """saves jsonl file"""
#     print(f"Saving to {save_path}")
#     with open(save_path, mode="w",encoding="utf-8") as opened_jsonl:
#         for json_dict in tqdm(dataset["train"]):
#             json.dump(json_dict, opened_jsonl)
#             opened_jsonl.write("\n")


def main():
    args = parse_cla()
    dataset = load_dataset()
    save_path=args.save_path
    dataset.save_to_disk(save_path)


if __name__ == "__main__":
    main()
