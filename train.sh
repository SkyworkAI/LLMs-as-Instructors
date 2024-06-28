for lr in 2e-6 5e-6 5e-7 
do 

for accu in 1 4 8 32

do 
   echo $accu
   echo $lr
   echo ----------------
   
OUTPUT_PATH=LLMs_as_Instructors/mistral_${lr}_${accu}_iteration1/
ZERO_STAGE=2
echo $OUTPUT_PATH
echo $ZERO_STAGE
rm -rf output/
mkdir -p $OUTPUT_PATH
model_name_or_path=train/model/Mistral-7b-instrucion/Mistral-7B-Instruct-v0.2

deepspeed --include=localhost:0,1,2,3 --master_port 5020 train/code/main.py \
   --sft_only_data_path result/mistral_LaI_LEC_raw_k_2_train_9000.jsonl \
   --eval_data_file result/mistral_LaI_LEC_raw_k_2_train_9000.jsonl \
   --data_split 10,0,0 \
   --model_name_or_path ${model_name_or_path} \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --max_seq_len 512 \
   --learning_rate $lr \
   --weight_decay 0. \
   --num_train_epochs 1 \
   --gradient_accumulation_steps ${accu} \
   --lr_scheduler_type cosine \
   --num_warmup_steps 100 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --output_dir $OUTPUT_PATH 


bash run_infer.sh mistral_${lr}_${accu}_iteraion1 > run_infer_mistral_${lr}_${accu}_iteration1.log

done 

done