import os
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
from loguru import logger
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import datasets
from dataclasses import dataclass

from typing import Optional, List, Dict,Literal,Type
from transformers.trainer_pt_utils import LabelSmoother
import os
from tqdm import tqdm
from transformers import (
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    DataCollatorForSeq2Seq,
)
from peft import (
    TaskType,
    LoraConfig,
    get_peft_model,
    PeftModel,
)
from accelerate import PartialState
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
    val_ratio:float = 0.1
    max_train_samples:int = 1000
    max_eval_samples:int = 100

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
def build_dataset(
        dataset_path_name:str,
        tokenizer:AutoTokenizer,
        num_proc:int,
        model_max_length: int,
        seed:int=42,
        val_ratio:float=0.1,

):
    '''prompt template:
        <|im_start|>system
        请回答以下法律问题:<|im_end|>
        <|im_start|>user
        如果被告人不服判决，有什么权利？<|im_end|>
        <|im_start|>assistant
        根据《刑事诉讼法》第294条，被告人或其近亲属不服判决的，有权向上一级人民法院上诉。辩护人经被告人或者其近亲属同意，也可以提出上诉。因此，被告人可以通过上诉的方式表达其对判决的不满。<|im_end|>

    '''
    assert os.path.exists(dataset_path_name)
    dataset = datasets.load_from_disk(dataset_path_name)
    dataset.shuffle(seed=seed)
    def process_function(examples):
        instruction=(f"<|im_start|>system\n请回答以下法律问题:<|im_end|>\n<|im_start|>user"
                     f"\n{examples['instruction']+examples['input']}<|im_end|>\n<|im_start|>assistant\n")
        instruction=tokenizer(instruction,add_special_tokens=False)
        response=tokenizer(f"{examples['output']}",add_special_tokens=False)
        input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.eos_token_id]
        attention_mask = instruction["attention_mask"] + response["attention_mask"] + [tokenizer.eos_token_id]
        labels = [IGNORE_INDEX] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.eos_token_id]
        if len(input_ids) > model_max_length:
            input_ids = input_ids[:model_max_length]
            attention_mask = attention_mask[:model_max_length]
            labels = labels[:model_max_length]
        result={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        return result

    assert val_ratio>=0 and val_ratio<=0.4
    split_dataset = dataset.train_test_split(
        test_size=val_ratio,
        seed=seed,
        shuffle=True
    )
    train_dataset = split_dataset["train"].map(
        process_function,
        num_proc=num_proc,
    )
    train_dataset=train_dataset.map(remove_columns=split_dataset["train"].column_names).filter(lambda x: len(x["input_ids"]) <= model_max_length,num_proc=num_proc)

    eval_dataset = split_dataset["test"].map(
        process_function,
        num_proc=num_proc
    )
    eval_dataset=eval_dataset.map(remove_columns=split_dataset["test"].column_names).filter(lambda x: len(x["input_ids"]) <= model_max_length,num_proc=num_proc)

    train_dataset=train_dataset.shuffle(seed=seed)
    eval_dataset=eval_dataset.shuffle(seed=seed)
    return {"train":train_dataset, "eval":eval_dataset}
def save_model(model, tokenizer, args):
    """Save the model and the tokenizer."""
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Take care of distributed/parallel training
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

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
    logger.info(f"lora config:{lora_args}")
    with PartialState().local_main_process_first():
        dataset_dict=build_dataset(data_args.dataset_path_name,tokenizer=tokenizer,val_ratio=data_args.val_ratio,
                      num_proc=data_args.num_proc,model_max_length=model_args.model_max_length,
                      )
        train_dataset=dataset_dict["train"]
        eval_dataset=dataset_dict["eval"]
        max_train_samples=min(len(train_dataset),data_args.max_train_samples)
        max_eval_samples=min(len(eval_dataset),data_args.max_eval_samples)
        train_dataset=train_dataset.select(range(max_train_samples))
        eval_dataset=eval_dataset.select(range(max_eval_samples))

    model = get_peft_model(model, lora_config)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=IGNORE_INDEX,
    )
    trainer=Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args
    )
    if trainer.is_world_process_zero():
        logger.info("training")
    train_results = trainer.train(
        resume_from_checkpoint=training_args.resume_from_checkpoint if training_args.resume_from_checkpoint else None
    )
    metrics = train_results.metrics
    metrics["train_samples"] = max_train_samples
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    if trainer.is_world_process_zero():
        logger.info(f"Training metrics: {metrics}")
        logger.info(f"Saving model checkpoint to {training_args.output_dir}")
        save_model(model, tokenizer, training_args)

    if trainer.is_world_process_zero():
        logger.info("evaluating")
    metrics_eval = trainer.evaluate(metric_key_prefix="eval")
    metrics_eval["eval_samples"] = max_eval_samples
    try:
        perplexity = math.exp(metrics_eval["eval_loss"])
    except OverflowError:
        perplexity = float("inf")
    metrics_eval["perplexity"] = perplexity
    trainer.log_metrics("eval", metrics_eval)
    trainer.save_metrics("eval", metrics_eval)
    if trainer.is_world_process_zero():
        logger.info(f"evaluation metrics: {metrics_eval}")

if __name__ == '__main__':
    main()