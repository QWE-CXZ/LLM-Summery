import argparse
import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    BitsAndBytesConfig,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', default=None, type=str, required=True)
    parser.add_argument('--cache_dir', default=None, type=str, )
    parser.add_argument('--lora_model', default="", type=str, help="If None, perform inference on the base model")
    parser.add_argument('--tokenizer_path', default=None, type=str)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument('--load_in_8bit', action='store_true', help='Whether to load model in 8bit')
    parser.add_argument('--load_in_4bit', action='store_true', help='Whether to load model in 4bit')
    args = parser.parse_args()
    print(args)
    load_type = 'auto'
    if args.tokenizer_path is None:
        args.tokenizer_path = args.base_model

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True, padding_side='left')
    config_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": load_type,
        "low_cpu_mem_usage": True,
        "device_map": 'auto',
        "cache_dir": args.cache_dir,
    }
    if args.load_in_8bit:
        config_kwargs['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True)
    elif args.load_in_4bit:
        config_kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=load_type,
        )
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, **config_kwargs)
    try:
        base_model.generation_config = GenerationConfig.from_pretrained(args.base_model, trust_remote_code=True)
    except OSError:
        print("Failed to load generation config, use default.")
    if args.lora_model:
        model = PeftModel.from_pretrained(base_model, args.lora_model, torch_dtype=load_type, device_map='auto')
        print("Loaded lora model")
    else:
        model = base_model
        print("Loaded raw model")
    model.eval()
    instruction="请回答以下法律问题:"
    question1="小明因涉嫌盗窃被公安机关拘留，拘留期限届满，但案件尚未办结，需要继续侦查。公安机关根据法规应该怎样办理？"
    question2="某人因涉嫌违规行为被公安机关拘留，在满足哪些条件下可以有担保人暂缓执行行政拘留？"
    question3="在驾驶机动车时，车辆前方遇到一辆停车排队或缓慢行驶的车辆，驾驶员不按规定避让并选择借道超车或者占用对面车道、穿插等候车辆，这样的行为是否违法？"
    question4 = "某公司向社会公开募集股份，签订了承销协议，并同银行签订了代收股款协议。股份发行后，股款已经缴足且验资机构已出具了证明。然而，发起人在股款缴足后三十日内未能召开创立大会，认股人想要返还所缴股款加上同期存款利息。他们能够这样要求发起人吗？"
    question5="某企业申请破产，其负债总额为500万元，其中包括债权人张先生的借款100万元，张先生与企业财务经理在6个月前私下协议，由企业提前对其借款进行清偿。此时，管理人可以对此进行何种请求？这种行为是否有效？"
    prompt=[question1,question2,question3,question4,question5]
    for i in range(len(prompt)):
        messages = [
            {
                "role": "system", "content": {instruction}},
            {
                "role": "user", "content": prompt[i]}
        ]

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to('cuda')

        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(f"prompt{i+1}:\n{prompt[i]}")
        print(f"response:\n{response}")

if __name__ == '__main__':
    main()