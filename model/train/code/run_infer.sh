# model_name_or_path=$1

# OUTPUT_PATH=../../train/code/LLMs_as_Instructors/$1/ 

# echo "Modified Model Path: $OUTPUT_PATH"

# bash ../../evaluation/run_shell/run_mmlu.sh $OUTPUT_PATH &
# bash ../../evaluation/run_shell/run_arc.sh $OUTPUT_PATH &
# bash ../../evaluation/run_shell/run_gsm8k.sh $OUTPUT_PATH &
# bash ../../evaluation/run_shell/run_gsm8k+.sh $OUTPUT_PATH &
# bash ../../evaluation/run_shell/run_mbpp.sh $OUTPUT_PATH &
# bash ../../evaluation/run_shell/run_humaneval.sh $OUTPUT_PATH &
# bash ../../evaluation/run_shell/run_bbh.sh $OUTPUT_PATH &

# wait  

# echo "Finished"


bash ../../evaluation/run_shell/run_mbpp.sh mistral_131431_34343_8000 &