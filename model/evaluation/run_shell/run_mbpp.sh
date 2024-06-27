model_name_or_path=$1


CUDA_VISIBLE_DEVICES=2 python ../../evaluation/get_performance_mbpp.py --model_name_or_path $1 &
CUDA_VISIBLE_DEVICES=2 python ../../evaluation/get_performance_mbpp.py --model_name_or_path $1 --train Train &


wait

modified_model_path=$(python ../../evaluation/run_shell/process_path.py $1)

echo "Modified Model Path: $modified_model_path"

bash ../../../result/mbpp/run_mbpp.sh $modified_model_path


