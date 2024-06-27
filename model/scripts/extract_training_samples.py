import json
import re
import argparse


def remove_prefix(s):
    prefixes = ["[QUESTION]\n", "[QUESTION]", "**Question:**", "Question:"]
    for prefix in prefixes:
        if s.startswith(prefix):
            return s[len(prefix):].lstrip()
    return s

def remove_last_double_newline(text):
    last_double_newline = text.rfind("\n\n")
    if last_double_newline != -1:
        text = text[:last_double_newline]
    return text


def extract_questions_and_answers_gsm8k(text):

    pattern = r"(?:(?:Question \d+|Sample \d+):|\*\*Question:\*\*|\[QUESTION\])\s*(.*?)\s*(?:(?:\*\*Answer:\*\*|\[ANSWER\]|Answer:))\s*(.*?)(?:\n####\s*(\d+))?(?=\s*(?:Question \d+|Sample \d+|###|$))"
    matches = re.findall(pattern, text, re.DOTALL)
    extracted_data = []
    
    for match in matches:
        question, answer, number = match
        if number:
            answer += f" The answer is {number}."
        extracted_data.append({
            "question": question.strip(),
            "answer": answer.strip()
        })
    return extracted_data



def extract_questions_and_codes_mbpp(text):

    pattern = r"### Question \d+:(.*?)(```python\n.*?```)"
    matches = re.findall(pattern, text, re.DOTALL)
    extracted_data = []
    
    for match in matches:
        question, code = match
        extracted_data.append({
            "question": question.strip(),
            "code": code.strip()
        })
    return extracted_data

def extract_questions_and_answers_mmlu(text):
    
    pattern = r"(?:Question \d+:|Question:|### Question \d+:|## Question \d+:)\s*(.*?)\s*(?:Answer:|\*\*Answer:\*\*)\s*(.*?)(?=\s*(?:Question \d+:|Question:|### Question \d+:|## Question \d+:|$))"

    matches = re.findall(pattern, text, re.DOTALL)
    extracted_data = []
    for match in matches:
        question, answer = match
        extracted_data.append({
            "question": question.strip(),
            "answer": answer.strip()
        })
    
    return extracted_data



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="gsm8k", type=str)
    parser.add_argument('--model', default="mistral", type=str, help="The model name")
    parser.add_argument('--model_setting', default="raw", type=str, help="Model configuration setting")
    parser.add_argument('--sample_size', default=3000, type=int, help="The size of the training data")
    args = parser.parse_args()

    file = []
    for line in open(f"../../result/{args.dataset}/{args.model}/{args.model_setting}.jsonl"):
        file.append(json.loads(line.strip()))

    print("<----------------extracting_training_samples--------------->")
    new_sample = []
    new_sample_q = []
    for iteration in file:
        for line in iteration:
            if args.dataset == "gsm8k":
                result = extract_questions_and_answers_gsm8k(line['reponse'])
                for sample in result:
                    question = remove_last_double_newline(remove_prefix(sample["question"]))
                    answer = sample["answer"]

                    if question != None and answer != None and question not in new_sample_q:
                        new_sample_q.append(question)
                        new_sample.append({"question":question, "response":answer})

            elif args.dataset == "mmlu":
                result = extract_questions_and_answers_mmlu(line['reponse'])
                for sample in result:
                    question = remove_last_double_newline(sample["question"])
                    answer = "Answer: " + remove_last_double_newline(sample["answer"])
                
                    if question != None and answer != None and question not in new_sample_q:
                        new_sample_q.append(question)
                        new_sample.append({"question":question, "response":answer})

            elif args.dataset == "mbpp":
                questions_and_codes = extract_questions_and_codes_mbpp(line['reponse'])
                for sample in questions_and_codes:
                    question = sample["question"]
                    code = sample["code"]

                    if question != None and code != None and question not in new_sample_q:
                        new_sample_q.append(question)
                        new_sample.append({"question":question, "response":code})


    # Check if the length of new_sample is less than the specified sample_size
    if len(new_sample) < args.sample_size:
        print("The number of new samples is less than the desired sample size.")
        print("It is recommended to run the LLMs-as-Instructors process again to generate more samples.")
    else:
        print(f"New training dataset {args.dataset} have {len(new_sample)} samples")
        out_dir = f"../../result/{args.dataset}/{args.model}/{args.model_setting}_train.jsonl"
        print(f"<--------------written in the file {out_dir}--------------->")
        json_data = json.dumps(new_sample, sort_keys= False, ensure_ascii=False, indent=4, separators=(',', ': '))
        file = open(out_dir, "w", encoding='utf-8')
        file.write(json_data)

    assert len(new_sample) >= args.sample_size, "Insufficient number of samples generated; consider re-running the generation process."