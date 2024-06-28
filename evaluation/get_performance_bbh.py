import torch
import json
import glob
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, AutoConfig
import argparse
from transformers import TextGenerationPipeline
from tqdm import tqdm
import json, os
import re
from thefuzz import process
import copy
import tensor_parallel as tp
import jsonlines

os_data = "data/bbh"
os_performance_result = "train/code/result/"
os_result = "result"

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path',default="llama3")
parser.add_argument("-b", "--batch-size", type = int, default = 1)
parser.add_argument("-t", "--train", type = str, default = "Test")
parser.add_argument("--base_model_is", type = str, default = "")
args = parser.parse_args()

print("model_name_or_path: " + args.model_name_or_path)


def get_input_sample(prompt, sample):
    question_sample = prompt.strip() + "\n\nQ: " + sample["input"]
    return "", question_sample


def format_tokens(samples):
    return_samples = []
    for sys, question in samples:
        if args.base_model_is == "llama3":
            return_sample = f'''<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n A:'''
        else:
            return_sample = f'''<s>[INST] {question}[/INST] A:'''
        return_samples.append(return_sample)

if __name__ == '__main__':

    if args.model_name_or_path == "llama3":
        args.model_name_or_path = "train/model/Meta-Llama-3-8B-Instruct"
        the_model_is = "llama3"
    elif args.model_name_or_path == "llama2":
        args.model_name_or_path = "train/model/llama-2-7b-chat/Llama-2-7b-chat-hf"
        the_model_is = "llama2"
    elif args.model_name_or_path == "mistral":
        args.model_name_or_path = "train/model/Mistral-7b-instrucion/Mistral-7B-Instruct-v0.2"
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
    count = 0
    all_tasks = {}
    all_prompts = {}
    task_files = glob.glob(os.path.join(os_data + "bbh/",  "*.json"))
    for task_file in tqdm(task_files, desc="Loading tasks"):
        task_name = ""
        with open(task_file, "r") as f:
            task_name = os.path.basename(task_file).split(".")[0]
            all_tasks[task_name] = json.load(f)["examples"]

        with open(os_data + f"cot-prompts/{task_name}.txt", "r") as f_prompt:
            task_prompt = "".join(f_prompt.readlines()[2:])
            all_prompts[task_name] = task_prompt

    for task_name in tqdm(all_tasks.keys(), desc="Evaluating"):
        file = all_tasks[task_name]
        prompt = all_prompts[task_name]
        print(f"<-------------------task:bbh--{task_name}----------------------->")
        for i in tqdm(range(0, len(file), args.batch_size), total = len(file) // args.batch_size):
            start_idx = i 
            end_idx = min(i + args.batch_size, len(file))
            batch_texts = file[start_idx:end_idx]
            batch_sample = [get_input_sample(prompt, sample) for sample in batch_texts]
            batch_sample = format_tokens(batch_sample)
            if args.base_model_is == "llama3":
                generator = TextGenerationPipeline(model = model, tokenizer = tokenizer)
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
                generator = TextGenerationPipeline(model = model, tokenizer = tokenizer)
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
                new_sample["response"] =  response
                extracted_answer = re.search(r"[t|T]he answer is (.*?)\.", response)
                if extracted_answer:
                    pred = extracted_answer.group(1).strip()
                else:
                    pre = response.strip()
                if pred.strip().lower() == new_sample["target"].lower():
                    acc += 1
                    out_dir = f"{os_result}/bbh/{args.base_model_is}/{args.train}_{the_model_is}_right.jsonl"
                    os.makedirs(os.path.dirname(out_dir), exist_ok=True)
                    with jsonlines.open(out_dir, 'a') as fw:
                        fw.write(new_sample)
                else:
                    
                    out_dir = f"{os_result}/bbh/{args.base_model_is}/{args.train}_{the_model_is}_wrong.jsonl"
                    os.makedirs(os.path.dirname(out_dir), exist_ok=True)
                    with jsonlines.open(out_dir, 'a') as fw:
                        fw.write(new_sample)
                count += 1

    avg_loss = acc / count
    out_dir = os_performance_result + f"{args.base_model_is}_result.txt"
    os.makedirs(os.path.dirname(out_dir), exist_ok=True)
    with open(out_dir, "a", encoding="utf8") as f:
        avg_loss = acc / len(file)
        f.write(f"{the_model_is}\tbbh\t{avg_loss:.4f}\n")
    print(f"<---------------model:{the_model_is}--task:bbh------acc:{avg_loss}--------------------->")