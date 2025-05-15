from pathlib import Path
from tqdm import tqdm
import json
import datasets
from typing import Dict, Type, List
import argparse


def parse_cla():
    """parses command-line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-save_path", type=Path)
    return parser.parse_args()


def dataset_dict(input:str, output:str) -> Dict:
    """

    keyword arguments:
    input -- User input text
    output -- Model output text
    """
    input_txt = f"### Instruction \n请回答以下法律问题 \n### Input \n{input}"
    output_txt = f"### Output {output}"
    return {"prompt": input_txt, "Output": output_txt}


def load_dataset() -> Type[datasets.Dataset]:
    """returns tldr dataset"""
    return datasets.load_dataset("Kuugo/chinese_law_ft_dataset",split='train',trust_remote_code=True)


def save_list(tldr_dataset:Type[datasets.Dataset]) -> List:
    """saves list of dataset dictionaries"""
    save_list = []
    for text_dict in tqdm(tldr_dataset["train"]):
        prompt = dataset_dict(input=text_dict["instruction"],  output=text_dict["output"])
        save_list.append(prompt)
    return save_list


def save_jsonl(save_list:List, save_path:Path):
    """saves jsonl file"""
    with open(save_path, mode="w") as opened_jsonl:
        for json_dict in save_list:
            json.dump(json_dict, opened_jsonl)
            opened_jsonl.write("\n")


def main():
    args = parse_cla()
    dataset = load_dataset()
    s_list = save_list(tldr_dataset=dataset)
    save_jsonl(save_list=s_list, save_path=args.save_path)


if __name__ == "__main__":
    main()