import torch
import textwrap
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, AutoConfig
import argparse
from transformers import TextGenerationPipeline
from tqdm import tqdm
import json, os
import math
import re
from thefuzz import process
import copy
import tensor_parallel as tp
import jsonlines

os_data = "../../../data/humaneval"
os_result = "../../train/code/result/"

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path',default="llama3")
parser.add_argument("-b", "--batch-size", type = int, default = 1)
parser.add_argument("-t", "--train", type = str, default = "Test")
parser.add_argument("-o", "--out", type = str, default = "")
parser.add_argument("--base_model_is", type = str, default = "")
args = parser.parse_args()

print("model_name_or_path: " + args.model_name_or_path)

def get_input_sample(sample):

    match = re.search(rf"def\s+({sample['entry_point']}.*?):\s*\n", sample["prompt"])

    if match:
        signature = match.group(1)
    else:
        signature = ""

    search_result = re.search(
    rf"(?:\"\"\"|''')(.*?)(?:\"\"\"|''')", sample["prompt"], re.DOTALL)

    if search_result is not None:
        description = "\n".join(
            [
                line.strip()
                for line in search_result.group(1).split("\n")
            ]
        )
    else:
        description = ""
    prompt = (
        f"Write a Python function `{signature}` to solve the following problem:\n"
        f"{description}\n"
        f"{sample['prompt']}"
    )
    return prompt


def extract_code(text, entry_point):

    code_block_pattern = re.compile(
        rf"```(?:[Pp]ython\n)?.*?def\s+{entry_point}.*?:\n(.*?)\n```", re.DOTALL
    )
    code_block = code_block_pattern.search(text)
    if code_block is None:
        code_block_pattern = re.compile(
            rf"def\s+{entry_point}.*?:\n(.*?)(?:\n(?!\n*(?:  |\t))|$)", re.DOTALL
        )
        code_block = code_block_pattern.search(text)
    if code_block is None:
        code_block_pattern = re.compile(
            r"def.*?:\n(.*?)(?:\n(?!\n*(?:  |\t))|$)", re.DOTALL
        )
        code_block = code_block_pattern.search(text)

    if code_block is not None:
        return code_block.group(1)

    return textwrap.indent(text, " " * 4)


def format_tokens(samples):
    return_samples = []
    for question in samples:
        if args.base_model_is == "llama3":
            return_sample = f'''<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'''
        else:
            return_sample = f'''<s>[INST] {question}[/INST]'''
        return_samples.append(return_sample)

    return return_samples

if __name__ == '__main__':

    if args.model_name_or_path == "llama3":
        args.model_name_or_path = "../../train/model/Meta-Llama-3-8B-Instruct"
        the_model_is = "llama3"
    elif args.model_name_or_path == "llama2":
        args.model_name_or_path = "../../train/model/llama-2-7b-chat/Llama-2-7b-chat-hf"
        the_model_is = "llama2"
    elif args.model_name_or_path == "mistral":
        args.model_name_or_path = "../../train/model/Mistral-7b-instrucion/Mistral-7B-Instruct-v0.2"
        the_model_is = "mistral"
    else:
        trimmed_path = args.model_name_or_path.rstrip('/')
        parts = trimmed_path.split('/')
        if parts[-1].isdigit():
            the_model_is = f"{parts[-2]}_{parts[-1]}"
        else:
            the_model_is = parts[-1]
    args.base_model_is = ""
    if "llama2" in the_model_is:
        args.base_model_is = "llama2"
    elif "llama3" in the_model_is:
        args.base_model_is = "llama3"
    elif "mistral" in the_model_is:
        args.base_model_is = "mistral"
    else:
        print("Model name error")

    print(f"<-------------------The model is {the_model_is}, base model is {args.base_model_is}------------->")

    load_type = torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side="left")
    print("Load tokenizer successfully")

    model_config = AutoConfig.from_pretrained(args.model_name_or_path)
    print("Load config successfully")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, 
        torch_dtype=load_type,
        config=model_config,
        device_map = "auto"
        ).eval()
    print("Load model successfully")
    torch.cuda.empty_cache()
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
   
    acc = 0
    file = []
    if args.train == "Train":
        print("Load training set successfully")
        file = json.load(open(os_data  + "/" + "train.json","r", encoding='utf-8'))
    else:
        for line in open(os_data  + "/" + "test.jsonl","r", encoding='utf-8'):
            file.append(json.loads(line.strip()))

    print(f"<-------------------task:humaneval-{args.train}----source{len(file)}-------------------->")
    for i in tqdm(range(0, len(file), args.batch_size), total = len(file) // args.batch_size):
        start_idx = i 
        end_idx = min(i + args.batch_size, len(file))
        batch_texts = file[start_idx:end_idx]
        batch_sample = [get_input_sample(sample) for sample in batch_texts]
        batch_sample = format_tokens(batch_sample)
        if args.base_model_is == "llama3":
            generator = TextGenerationPipeline(model = model, tokenizer = tokenizer, batch_size=args.batch_size)
            outputs = generator(
                batch_sample, 
                pad_token_id = tokenizer.eos_token_id,
                eos_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>"),
                do_sample = False, 
                max_new_tokens = 1024,
                return_full_text = False, 
                top_p = 1.0,
                temperature = 0.0
            )
        else:
            generator = TextGenerationPipeline(model = model, tokenizer = tokenizer, batch_size=args.batch_size)
            outputs = generator(
                batch_sample, 
                pad_token_id = tokenizer.eos_token_id,
                do_sample = False, 
                max_new_tokens = 1024,
                return_full_text = False, 
                top_p = 1.0,
                temperature = 0.0
            )
    
        for index, sample in enumerate(batch_texts):
            response = outputs[index][0]['generated_text']
            new_sample = copy.deepcopy(batch_texts[index])
            answer = extract_code(response, new_sample["entry_point"])
            new_sample["completion"] =  answer
            new_sample["response"] =  response

            out_dir = f"{os_result}/humaneval/{args.train}_{the_model_is}.jsonl"
            os.makedirs(os.path.dirname(out_dir), exist_ok=True)
            with jsonlines.open(out_dir, 'a') as fw:
                fw.write(new_sample)
