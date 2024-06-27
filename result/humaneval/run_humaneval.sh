source activate codex

model_is=$1

echo "Testing Model Humaneval: $1"

evaluate_functional_correctness ../../../result/humaneval/Test_$1.jsonl --problem_file=../../../data/humaneval/test.jsonl


python ../../../result/humaneval/process_result.py --model_is $1