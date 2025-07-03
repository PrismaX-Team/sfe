export HF_HOME=
export HF_TOKEN=
export OPENAI_API_KEY=
export OPENAI_API_BASE=

export API_TYPE="openai"
export MODEL_VERSION="gpt-4o-2024-11-20"

# export AZURE_OPENAI_API_KEY=""
# export AZURE_OPENAI_API_BASE=""
# export AZURE_OPENAI_API_VERSION="2023-07-01-preview"

# pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git

# GPT Series
export FILE_NAME="lmms_eval_gpt4o_en.json"
python3 -m lmms_eval \
    --model openai_compatible \
    --model_args model_version=gpt-4o-2024-11-20,azure_openai=False \
    --tasks sfe-en \
    --batch_size 1

export FILE_NAME="lmms_eval_gpt4o_zh.json"
python3 -m lmms_eval \
    --model openai_compatible \
    --model_args model_version=gpt-4o-2024-11-20,azure_openai=False \
    --tasks sfe-zh \
    --batch_size 1