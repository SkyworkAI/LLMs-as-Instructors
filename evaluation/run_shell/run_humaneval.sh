model_name_or_path=$1


CUDA_VISIBLE_DEVICES=1 python evaluation/get_performance_humaneval.py --model_name_or_path $1 &


modified_model_path=$(python evaluation/run_shell/process_path.py $1)

echo "Modified Model Path: $modified_model_path"

bash result/humaneval/run_humaneval.sh $modified_model_path
