source activate codex

model_is=$1

echo "Testing Model MBPP: $1"

evaluate_functional_correctness ../../../result/mbpp/Train_$1.jsonl --problem_file=../../../data/mbpp/train_refined_data.jsonl

evaluate_functional_correctness ../../../result/mbpp/Test_$1.jsonl --problem_file=../../../data/mbpp/test_refined_data.jsonl

python ../../../result/mbpp/process_result.py --model_is $1



