import os
import time
import copy
import openai
import json
import sys
import random
import time
import math
import re
import argparse
import backoff 
import jsonlines
from tqdm import tqdm
import concurrent.futures

openai.api_key = os.getenv('OPENAI_API_KEY')
@backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.Timeout, openai.error.APIError, openai.error.ServiceUnavailableError))
def llm(message):
    response = openai.ChatCompletion.create(
    model="gpt-4-0125-preview",
    messages = message,
    temperature = 0.8,
    max_tokens = 4095
    )
    return response['choices'][0]['message']['content']


def process(sample):
    """
    Creates a message from the sample data and sends it to the LLM function
    """
    message = [{"role": "system", "content": sample["prompt"]},
                {"role": "user", "content": sample["question"]}]
    response = llm(message)
    generate_new_sample = copy.deepcopy(sample)
    generate_new_sample["reponse"] = response

    return generate_new_sample


def parallel_calls(messages):
    """
    Executes the process function in parallel using threads
    """
    futures = []
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for message in messages:
            future = executor.submit(process, message)
            futures.append(future)
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing messages"):
            results.append(future.result())
    return results

def get_input_sample(sample):
    """
    Parses a code definition and description from a given sample for better display
    """
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="gsm8k", type=str)
    parser.add_argument('--model', default="mistral", type=str, help="The model name")
    parser.add_argument('--model_setting', default="raw", type=str, help="Model configuration setting")
    parser.add_argument('--sample_size', default=3000, type=int, help="The size of the training data")

    args = parser.parse_args()

    # Loads the incorrect samples from a file
    file_wrong  = []
    for line in open(f"../../result/{args.dataset}/{args.model}/Train_{args.model_setting}_wrong.jsonl", "r"):
        file_wrong.append(json.loads(line.strip()))
    # To guarantee the size of the generated samples we plus 1 for each sample based generation
    generated_samples_iteration = math.ceil(args.sample_size / len(file_wrong)) + 1
    message_list = []
    prompt = open(f"../../data/prompt/LLMs_as_Instructors/{args.dataset}.txt").read()

    # Generation of new questions or tasks based on wrong samples
    for idx, sample in enumerate(file_wrong):
        user_question = ""
        if args.dataset == "mbpp":
            input_sample = {"question":get_input_sample(sample), "wrong response": sample["completion"]}
            user_question = "The student was given the following coding writing question:\n\n[QUESTION]" + input_sample["question"] + \
                "\n\nThe student's wrong code answer, which failed to pass the test cases, to the question is \n\n[WRONG ANSWER]" + input_sample["wrong response"] + \
                f'''\n\nPlease follow the requirements and generate {generated_samples_iteration} similar code questions, along with the correct code and annotation. '''
        elif args.dataset == "mmlu":
            question = sample["question"] + "\n"
            for choice in ["A", "B", "C", "D"]:
                question += f'{choice}. {sample[f"{choice}"]}\n'
            input_sample = {"question":question, "wrong response": sample["response"]}
            user_question = "The following is a multiple choice question that the student got wrong including the correct answer from the answer sheet:\n\n[QUESTION]" + input_sample["question"] + \
                "\n\nThe student's wrong answer: \n\n[WRONG ANSWER]" + input_sample["wrong response"] + \
                f'''\n\nPlease follow the requirements and generate {generated_samples_iteration} similar questions, along with 4 different holding options, the correct answer and the brief explaination.'''
        else:
            input_sample = {"question":sample["question"], "wrong response": sample["response"]}
            user_question = "The student was given the following question:\n\n[QUESTION]" + input_sample["question"] + \
                "\n\nThe student's wrong response is \n\n[WRONG ANSWER]" + input_sample["wrong response"] + \
                f'''\n\nPlease follow the requirements and generate {generated_samples_iteration} similar question, along with the correct calculations and answer.'''
            
        message_list.append({"prompt":prompt, "question":user_question})

    # Parallel calls the openai API
    final_messages = parallel_calls(message_list)
    with jsonlines.open(f"../../data/{args.dataset}/{args.model}/LaI_{args.model_setting}.jsonl", 'a') as fw:
        fw.write(final_messages)
    



  