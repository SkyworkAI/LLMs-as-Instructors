import os
import openai
import json
import sys
import random
import time
import re
import numpy as np
import argparse
import backoff 
import jsonlines
import time
import faiss
import math
import copy
from tqdm import tqdm
import concurrent.futures
from collections import defaultdict
from sklearn.preprocessing import normalize
from transformers import BertTokenizer, BertModel



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


def sentence_to_vec(sentence, tokenizer, model, max_length=512):
    """
    Converts the query into a vector representation using a bert model.
    
    Returns:
        numpy.ndarray: A numpy array representing the average of the last hidden state vectors of the input sentence.
    """
    inputs = tokenizer(sentence, return_tensors="pt", max_length=max_length, truncation=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

def search(query_vector, k=5):
    """
    Search for the nearest k vector
    
    Returns:
        The distance D and index I.
    """
    D, I = index.search(query_vector.reshape(1, -1), k)  
    return D, I  

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="gsm8k", type=str)
    parser.add_argument('--model', default="mistral", type=str, help="The model name")
    parser.add_argument('--model_setting', default="raw", type=str, help="Model configuration setting")
    parser.add_argument('--sample_size', default=3000, type=int, help="The size of the training data")
    parser.add_argument('--k', default=3, type=int, help="The number of the nearest sample")
    args = parser.parse_args()

    # Loads the correct samples from the file and vectorize
    file_right  = []
    for line in open(f"result/{args.dataset}/{args.model}/Train_{args.model_setting}_right.jsonl", "r"):
        file_right.append(json.loads(line.strip()))
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    sentence_vectors = []
    if args.dataset == "mbpp":
        sentence_vectors = np.vstack([sentence_to_vec(sentence["prompt"], tokenizer, model) for sentence in file_right])
    else:
        sentence_vectors = np.vstack([sentence_to_vec(sentence["question"], tokenizer, model) for sentence in file_right])
    sentence_vectors = normalize(sentence_vectors, norm='l2')
    dimension = sentence_vectors.shape[1]  
    index = faiss.IndexFlatL2(dimension)  
    index.add(sentence_vectors) 


    # Loads the incorrect samples from the file and vectorize
    file_wrong  = []
    file_query = []
    question_set = []
    for line in open(f"result/{args.dataset}/{args.model}/Train_{args.model_setting}_wrong.jsonl", "r"):
        file_wrong.append(json.loads(line.strip()))
        if args.dataset == "mbpp":
            question_set.append(json.loads(line.strip())["prompt"])
        else:
            question_set.append(json.loads(line.strip())["question"])

    # Using vectorized incorrect samples to retrieval k nearest correct sample
    for sample in tqdm(file_wrong):
        if args.dataset == "mbpp":
            query = sentence_to_vec(sample["prompt"], tokenizer, model) 
        else:
            query = sentence_to_vec(sample["question"], tokenizer, model) 
        query = normalize(query, norm='l2')  
        distances, indices = search(query, k=args.k)
        closest_sentences = [file_right[idx] for idx in indices[0]]
        corrct_query = []
        for re_sample in closest_sentences:
            re_question = ""
            if args.dataset == "mbpp":
                re_question = re_sample["prompt"]
            else:
                re_question = re_sample["question"]
            question_set.append(re_question)
            corrct_query.append(re_sample)
        sample["correct_queries"] = corrct_query
        file_query.append(sample)
    
    # Generation of new questions or tasks based on contrastive set
    generated_samples_iteration = math.ceil(args.sample_size / len(file_wrong)) + 1
    mmlu_task_subject = {}
    message_list = []
    message = []
    prompt = open(f"data/prompt/LLMs_as_Instructors/{args.dataset}.txt").read()
    for idx, sample in enumerate(file_query):
        user_question = ""
        if args.dataset == "mbpp":
            correct_questions = ""
            for query in sample["correct_queries"]:
                correct_questions += "[QUESTION]" + get_input_sample(query) + "\n\n"
            input_sample = {"question":get_input_sample(sample), "wrong response": sample["completion"]}
            user_question = "The student correctly answer the following question:\n\n" + correct_questions + \
                "While the student can not correcly answer:\n\n[QUESTION]" + input_sample["question"] + \
                "\n\nThe student's wrong code answer, which failed to pass the test cases, to the question is \n\n[Student's WRONG ANSWER]" + input_sample["wrong response"] + \
                f'''\n\nPlease follow the requirements and generate {generated_samples_iteration} similar code questions, along with the correct code and annotation. '''
            
        elif args.dataset == "mmlu":
            correct_questions = ""
            question = sample["question"] + "\n"
            for choice in ["A", "B", "C", "D"]:
                question += f'{choice}. {sample[f"{choice}"]}\n'
            for query in sample["correct_queries"]:
                wrong_question = query["question"] + "\n"
                for choice in ["A", "B", "C", "D"]:
                    wrong_question += f'{choice}. {query[f"{choice}"]}\n'
                correct_questions += "[QUESTION]" + wrong_question + "\n\n"
            input_sample = {"question":question, "wrong response": sample["response"]}
            user_question = "The student correctly answer the following question:\n\n" + correct_questions + \
                "While the following is the question that the student got wrong:\n\n[QUESTION]" + input_sample["question"] + \
                "\n\nThe student's wrong answer: \n\n[Student's WRONG ANSWER]" + input_sample["wrong response"] + \
                f'''\n\nPlease follow the requirements and generate {generated_samples_iteration} similar questions, along with 4 different holding options, the correct answer and the brief explaination.'''
        else:
            input_sample = {"question":sample["question"], "wrong response": sample["response"]}
            correct_questions = ""
            for query in sample["correct_queries"]:
                correct_questions += "[QUESTION]" + query["question"] + "\n\n"
            user_question = "The student correctly answer the following question:\n\n" + correct_questions + \
                "\nWhile the student can not correcly answer the question:\n\n[QUESTION]" + input_sample["question"] + \
                "\n\n[Student's WRONG ANSWER]" + input_sample["wrong response"] + \
                f'''\n\nPlease follow the requirements and generate {generated_samples_iteration} similar questions, along with the correct calculations and answer.'''
            
        message_list.append({"prompt":prompt, "question":user_question})

    # Parallel calls the openai API
    final_messages = parallel_calls(message_list)
    with jsonlines.open(f"result/{args.dataset}/{args.model}/LaI_LEC_{args.model_setting}_k_{args.k}.jsonl", 'a') as fw:
        fw.write(final_messages)
    



  