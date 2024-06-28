model_name_or_path=$1

CUDA_VISIBLE_DEVICES=1 python evaluation/get_performance_arc.py --model_name_or_path $1 

wait



