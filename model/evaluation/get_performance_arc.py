import torch
import json
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

os_data = "../../../data/arc"
os_result = "../../train/code/result/"

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path',default="llama3")
parser.add_argument("-b", "--batch-size", type = int, default = 1)
parser.add_argument("-t", "--train", type = str, default = "Test")
parser.add_argument("--base_model_is", type = str, default = "")
args = parser.parse_args()

print("model_name_or_path: " + args.model_name_or_path)


def get_input_sample(sample):
    question_sample = sample["question"] + "\n"
    choices = sample["label"]
    for choice in choices:
        question_sample += f'{choice}. {sample[f"{choice}"]}\n'
    return "", question_sample

def extract_choice(gen, choice_list, choices):

    res = re.search(
        r"(?:(?:[Cc]hoose)|(?:(?:[Aa]nswer|[Cc]hoice)(?![^ABCD]{0,20}?(?:n't|not))[^ABCD]{0,10}?\b(?:|is|:|be))\b)[^ABCD]{0,20}?\b(A|B|C|D)\b",
        gen,
    )

    if res is None:
        res = re.search(
            r"\b(A|B|C|D)\b(?![^ABCD]{0,8}?(?:n't|not)[^ABCD]{0,5}?(?:correct|right))[^ABCD]{0,10}?\b(?:correct|right)\b",
            gen,
        )

    if res is None:
        res = re.search(r"^(A|B|C|D)(?:\.|,|:|$)", gen)

    if res is None:
        res = re.search(r"(?<![a-zA-Z])(A|B|C|D)(?![a-zA-Z=])", gen)

    if res is None:
        return choices[choice_list.index(process.extractOne(gen, choice_list)[0])]
    return res.group(1)

def format_tokens(samples):
    return_samples = []
    for sys, question in samples:
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
        file = json.load(open(os_data  + "/" + "train.jsonl","r", encoding='utf-8'))
    else:
        print("Load testing set successfully")
        file = json.load(open(os_data  + "/" + "test.jsonl","r", encoding='utf-8'))
    
    print(f"<-------------------task:arc--{args.train}---source{len(file)}-------------------->")
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
                pad_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>"),
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
            new_sample["response"] =  response
            pred = extract_choice(response, [sample[choice] for choice in sample["label"]], sample["label"])
            if pred == new_sample["answer"]:
                acc += 1 
                out_dir = f"{os_result}/arc/{args.base_model_is}/{args.train}_{the_model_is}_right.jsonl"
            os.makedirs(os.path.dirname(out_dir), exist_ok=True)
            with jsonlines.open(out_dir, 'a') as fw:
                fw.write(new_sample)
        else:
            out_dir = f"{os_result}/arc/{args.base_model_is}/{args.train}_{the_model_is}_wrong.jsonl"
            os.makedirs(os.path.dirname(out_dir), exist_ok=True)
            with jsonlines.open(out_dir, 'a') as fw:
                fw.write(new_sample)
       
    avg_loss = acc / len(file)
    out_dir = os_result + f"{args.base_model_is}_result.txt"
    os.makedirs(os.path.dirname(out_dir), exist_ok=True)
    with open(out_dir, "a", encoding="utf8") as f:
        avg_loss = acc / len(file)
        f.write(f"{the_model_is}\tarc\t{args.train}\t{avg_loss:.4f}\n")
    print(f"<---------------model:{the_model_is}--task:arc------acc:{acc/len(file)}--------------------->")