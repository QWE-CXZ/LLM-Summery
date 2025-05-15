from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import datasets
from dataclasses import dataclass

from typing import Optional, List, Dict
from accelerate import Accelerator
from transformers.trainer_pt_utils import LabelSmoother
import os
from typing import Literal
from tqdm import tqdm
from transformers import Trainer, TrainingArguments
from transformers.data.data_collator import DataCollatorWithFlattening
from datasets import Dataset,concatenate_datasets

IGNORE_INDEX = LabelSmoother.ignore_index
try:
    local_rank = int(os.environ["LOCAL_RANK"])
except:
    local_rank = 0
@dataclass
class ModelArguments:
    model_name_or_path: str = "Qwen2.5-1.5B"
    tokenizer_name: Optional[str] = "Qwen2.5-1.5B"
    cache_dir: Optional[str] = "Qwen_model_file"
    model_max_length: Optional[int] = 1024
    use_fast_tokenizer: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    trust_remote_code: bool = True
    rope_scaling: Optional[Literal["linear", "dynamic"]]=None
    flash_attn: bool = False
    shift_attn: bool = False
    neft_alpha: Optional[float] = None
    attn_implementation:str="flash_attention_2"

@dataclass
class DataArguments:
    dataset_path_name: str = "webis/tldr-17"
    num_proc:int = 4


def load_model(model_name:str):
    model=AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=ModelArguments.load_in_8bit,
        load_in_4bit=ModelArguments.load_in_4bit,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=ModelArguments.attn_implementation,
    )
    model.config.use_cache = False
    model.train()
    return model

def build_instruction_dataset(dataset_path_name:str, num_proc:int):
    assert os.path.exists(dataset_path_name)
    dataset = datasets.load_dataset(dataset_path_name)
    dataset = dataset.map(
        lambda x: {
            "input": f"### Instruction \nWrite a concise summary of the following text \n### Input \n{x['content']}",
            "output": f"### Output {x['summary']}"
        },
        num_proc=num_proc,
        remove_columns=["content", "summary"],
    )
    return dataset
