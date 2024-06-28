model_name_or_path=$1


CUDA_VISIBLE_DEVICES=3 python evaluation/get_performance_gsm8k+.py --model_name_or_path $1 

wait