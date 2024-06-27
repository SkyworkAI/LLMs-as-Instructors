import os
import json
import argparse

os_result = "../../../result/humaneval/"
os_out = "result/"

def main(model_is):
    
    os.makedirs(f'{os_result}result', exist_ok=True)
    
    for prefix in ['Test']:

        file_path = f"{os_result}{prefix}_{model_is}.jsonl_results.jsonl"
        correct_samples = []
        incorrect_samples = []

        try:
            with open(file_path, 'r') as file:
                for line in file:
                    data = json.loads(line.strip())
                    if data["passed"]:
                        correct_samples.append(data)
                    else:
                        incorrect_samples.append(data)
        except FileNotFoundError:
            print(f"Error: File {file_path} not found.")
            continue

        if "llama2" in args.model_is:
            out_base_model_is = "llama2"
        elif "llama3" in args.model_is:
            out_base_model_is = "llama3"
        elif "mistral" in args.model_is:
            out_base_model_is = "mistral"

        with open(f'{os_result}/{out_base_model_is}/{prefix}_{model_is}_right.jsonl', 'w') as correct_file:
            for sample in correct_samples:
                correct_file.write(json.dumps(sample) + '\n')

        with open(f'{os_result}/{out_base_model_is}/{prefix}_{model_is}_wrong.jsonl', 'w') as incorrect_file:
            for sample in incorrect_samples:
                incorrect_file.write(json.dumps(sample) + '\n')
        
        avg_loss = len(correct_samples) / (len(correct_samples) + len(incorrect_samples))
        out_dir = os_out + f"{out_base_model_is}_result.txt"
        os.makedirs(os.path.dirname(out_dir), exist_ok=True)
        with open(out_dir, "a", encoding="utf8") as f:
            avg_loss = avg_loss
            f.write(f"{model_is}\thumaneval {prefix}\t{avg_loss:.4f}\n")
        
        os.remove(file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process results and split into correct and incorrect samples.")
    parser.add_argument('--model_is', required=True, help="Model identifier used to specify the files.")
    parser.add_argument("--base_model_is", type = str, default = "")
    args = parser.parse_args()
    
    main(args.model_is)
