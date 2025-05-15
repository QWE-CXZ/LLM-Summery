import os
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import datasets
from dataclasses import dataclass

from typing import Optional, List, Dict,Literal,Type
from accelerate import Accelerator
from transformers.trainer_pt_utils import LabelSmoother
import os
from tqdm import tqdm
from transformers import (
    Trainer,
    TrainingArguments,
    HfArgumentParser,
)
from peft import (
    TaskType,
    LoraConfig,
    get_peft_model,
    PeftModel,
)
from transformers.data.data_collator import DataCollatorWithFlattening
from datasets import Dataset,concatenate_datasets

is_flash_attn_2_available = False
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input, unpad_input

    is_flash_attn_2_available = True
except ImportError:
    is_flash_attn_2_available = False

IGNORE_INDEX = LabelSmoother.ignore_index
try:
    local_rank = int(os.environ["LOCAL_RANK"])
except:
    local_rank = 0
@dataclass
class ModelArguments:
    model_name_or_path: str = "Qwen/Qwen2.5-1.5B"
    tokenizer_name_or_path: Optional[str] = "Qwen/Qwen2.5-1.5B"
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
    dataset_path_name: str = "sft_data"
    num_proc:int = 4

@dataclass
class TrainingArguments(TrainingArguments):
    output_dir = "./output/Qwen2.5-1.5B"
    per_device_train_batch_size:int = 4
    per_device_eval_batch_size:int = 4
    gradient_accumulation_steps:int = 4
    logging_steps:int = 10
    num_train_epochs:int = 3
    save_steps:int = 100
    learning_rate:float = 1e-4
    gradient_checkpointing:bool = True
@dataclass
class LoraArguments:
    task_type = TaskType.CAUSAL_LM
    target_model:str="all"
    r:int=16
    lora_alpha:int=32
    lora_dropout:float=0



def load_model(model_args:ModelArguments):
    if is_flash_attn_2_available:
        model=AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            load_in_8bit=model_args.load_in_8bit,
            load_in_4bit=model_args.load_in_4bit,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation=model_args.attn_implementation,
            cache_dir=model_args.cache_dir,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            load_in_8bit=model_args.load_in_8bit,
            load_in_4bit=model_args.load_in_4bit,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            cache_dir=model_args.cache_dir,
        )
    model.config.use_cache = False
    model.train()
    return model
def find_all_linear_names(peft_model, int4=False, int8=False):
    """Find all linear layer names in the model. reference from qlora paper."""
    cls = torch.nn.Linear
    if int4 or int8:
        import bitsandbytes as bnb
        if int4:
            cls = bnb.nn.Linear4bit
        elif int8:
            cls = bnb.nn.Linear8bitLt
    lora_module_names = set()
    for name, module in peft_model.named_modules():
        if isinstance(module, cls):
            # last layer is not add to lora_module_names
            if 'lm_head' in name:
                continue
            if 'output_layer' in name:
                continue
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return sorted(lora_module_names)
def build_instruction_dataset(dataset_path_name:str, num_proc:int):
    assert os.path.exists(dataset_path_name)
    dataset = datasets.load_dataset(dataset_path_name)
    
    
    return dataset

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, LoraArguments))
    model_args, data_args, training_args,lora_args = parser.parse_args_into_dataclasses()
    #print(training_args)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name_or_path,
    )

    model = load_model(model_args)
    lora_args.target_model = find_all_linear_names(peft_model=model,int4=model_args.load_in_4bit,int8=model_args.load_in_8bit)
    lora_config=LoraConfig(
        task_type=lora_args.task_type,
        target_modules=lora_args.target_model,
        r=lora_args.r,
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
        inference_mode=False
    )
    print(f"lora config:{lora_args}")
    model=get_peft_model(model,lora_config)
if __name__ == '__main__':
    main()