# Set the OpenAI API key as an environment variable
export OPENAI_API_KEY='your_api_key_here'
# Run the Python script with specific arguments:
# --model specifies the model to use, default is 'mistral'
# --model_setting sets the model configuration, default is 'raw'
# --dataset defines which dataset to use, options include 'gsm8k', 'mbpp', and 'mmlu'
# --sample_size sets the number of samples to process, here it is set to 3000

# Define variables for each parameter
MODEL="mistral"
MODEL_SETTING="raw"
DATASET="gsm8k"
SAMPLE_SIZE=3000

python LLMs_as_Instructors/LE.py \
    --model $MODEL \
    --model_setting $MODEL_SETTING \
    --dataset $DATASET \
    --sample_size $SAMPLE_SIZE

python scripts/extract_training_samples.py \
    --model $MODEL \
    --model_setting "LaI_LE_${MODEL_SETTING}" \
    --dataset $DATASET \
    --sample_size $SAMPLE_SIZE \

