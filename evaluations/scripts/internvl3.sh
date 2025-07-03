export HF_HOME=
export HF_TOKEN=
export OPENAI_API_KEY=
export OPENAI_API_BASE=

export API_TYPE="openai"
export MODEL_VERSION="gpt-4o-2024-11-20"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export VLLM_WORKER_MULTIPROC_METHOD=spawn
# export NCCL_BLOCKING_WAIT=1
# export NCCL_TIMEOUT=18000000
# export NCCL_DEBUG=DEBUG

# pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
export FILE_NAME="lmms_eval_internvl3-78b_en.json"
python3 -m lmms_eval \
    --model vllm \
    --model_args model_version=<MODEL_PATH>,tensor_parallel_size=8 \
    --tasks sfe-en \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix vllm

export FILE_NAME="lmms_eval_internvl3-78b_zh.json"
python3 -m lmms_eval \
    --model vllm \
    --model_args model_version=<MODEL_PATH>,tensor_parallel_size=8 \
    --tasks sfe-en \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix vllm

