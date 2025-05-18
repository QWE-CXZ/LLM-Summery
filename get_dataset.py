import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
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

def dataset_filter(dataset: datasets.Dataset) -> datasets.Dataset:
    def filt_by_law(example):
        ins_len = len(example['instruction']) + len(example['input'])
        if ins_len > 500:
            return False
        splited = example['output'].split("《")
        if len(splited) > 2 or len(splited) < 2:
            return False
        law = splited[1]
        if "刑法" not in law:
            return False
        if "修正" in law:
            return False
        return True
    dataset=dataset.filter(filt_by_law)
    return dataset
def main():
    args = parse_cla()
    dataset = load_dataset()
    save_path=args.save_path
    print(dataset['train'][:10])
    dataset=dataset['train']
    dataset=dataset_filter(dataset)
    dataset=dataset.select(range(12000))
    dataset.save_to_disk(save_path)


if __name__ == "__main__":
    main()
