import json
import random
import argparse


def convert_string_to_list(string):
    # Convert the comma-separated string back into a list
    return string.split(',')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="mistral", type=str, help="The model name")
    parser.add_argument('--dataset_list', default="gsm8k", type=str, required=True, help="A list of datasets name")
    parser.add_argument('--model_setting', default="raw", type=str, required=True, help="Model configuration setting")
    parser.add_argument('--sample_size_list', default="3000", type=str, required=True, help="A list of the training data")
    args = parser.parse_args()

    dataset_list = convert_string_to_list(args.dataset_list)
    sample_size_list = convert_string_to_list(args.sample_size_list)

    assert len(dataset_list) == len(sample_size_list), "Incorrect Parameter"

    merged_train = []
    for idx, dataset in enumerate(dataset_list):
        file = json.load(open(f'''result/{dataset}/{args.model}/{args.model_setting}.jsonl''', "r"))
        limit = int(sample_size_list[idx])
        if limit > len(file):
            limit = len(file)
        sampled_data = random.sample(file, limit)
        for sample in sampled_data: merged_train.append(sample)

    print(f"Finall training dataset {dataset} have {len(merged_train)} samples")
    random.shuffle(merged_train)
    file_path = f'''result/{args.model}_{args.model_setting}_{len(merged_train)}.jsonl'''
    print(f"Finall training dataset are written at {file_path}")
    json_data = json.dumps(merged_train, sort_keys= False, ensure_ascii=False, indent=4, separators=(',', ': '))
    file = open(file_path, "w", encoding='utf-8')
    file.write(json_data)