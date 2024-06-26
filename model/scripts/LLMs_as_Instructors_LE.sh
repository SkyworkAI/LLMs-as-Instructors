# Set the OpenAI API key as an environment variable
export OPENAI_API_KEY='your_api_key_here'


# Run the Python script with specific arguments:
# --model specifies the model to use, default is 'mistral'
# --model_setting sets the model configuration, default is 'raw'
# --dataset defines which dataset to use, options include 'gsm8k', 'mbpp', and 'mmlu'
# --sample_size sets the number of samples to process, here it is set to 3000
python ../instruction_get/LLMs_as_Instructors_LE.py \
    --model mistral \
    --model_setting raw \
    --dataset gsm8k \
    --sample_size 3000


