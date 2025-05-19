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
    question1="请根据基本案情，给出适用的法条。基本案情：经审理查明，2017年6月7日20时许，在长春市绿园区西新镇开元村小东沟屯吕某某、王桂荣家东屋，因琐事产生争执后，被告人王桂荣用手将被害人户某某推倒在地，致户某某右股骨粉碎性骨折。经长春市司法鉴定中心鉴定：户某某外伤致右股骨粉碎性骨折构成轻伤一级。2017年8月9日，民警在长春市绿园区西新镇开元村小东沟屯王桂荣家将王桂荣传唤到派出所。上述事实，被告人王桂荣在开庭审理过程中无异议，并有被告人王桂荣在侦查机关的供述、被害人户某某的陈述、证人于某某的证言、受案登记表及立案决定书、到案经过、户籍证明、指认现场笔录及照片、长春市公安司法鉴定中心法医学人体损伤程度鉴定意见书等证据证实，足以认定。"
    question2="以下情况是否属于刑事犯罪？一名人员在公共场合醉酒滋事，损坏公共财物。"
    question3="小明在街头摆摊贩卖一些明显是盗版的电影光盘，属于侵犯著作权行为吗？"
    question4="某政府部门的工作人员在办理某企业的审批手续时，向该企业索要了一定财物，以此为由加快审批进度。根据我国《刑法》相关规定，这名工作人员是否构成受贿罪？"
    question5="李某是一名国家工作人员，在履行公务期间窃取了国家机密并逃往了美国，如何处罚他？"
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
