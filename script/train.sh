# 需要配置tf_logging_dir, wandb_mode=offline

export tf_logging_dir=/home/admin/logs/tfevent
mkdir -p $tf_logging_dir
export WANDB_MODE="offline"

mkdir -p $output_dir
log_file=$output_dir/log-$NODE_NAME.txt
pip install openai
cd ../code/ms-swift-rlfkv
model_type='qwen3'
model_dir='qwen3-8b'
output_dir='output/qwen3_8b_rlfkv'
dataset='../dataset/rl_data_with_unit_number.json' 
pip install wandb
NNODES=${WORLD_SIZE:-1} \
NODE_RANK=${RANK:-0}  \
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1} \
RANDOM_PORT=$(shuf -i 20000-60000 -n 1)
MASTER_PORT=${MASTER_PORT:-$RANDOM_PORT} \
NPROC_PER_NODE=8 \
PYTORCH_CUDA_ALLOC_CONF='' \
SEQUENCE_PARALLEL_IMPL=ring_attention \
swift rlhf \
    --reward_funcs RLFKV \
    --external_plugins examples/train/grpo/plugin/plugin.py \
    --rlhf_type grpo \
    --model $model_dir \
    --model_type $model_type \
    --output_dir $output_dir \
    --dataset $dataset \
    --train_type full \
    --torch_dtype bfloat16 \
    --use_vllm true \
    --vllm_mode colocate \
    --sleep_level 1 \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_max_model_len 20000 \
    --vllm_tensor_parallel_size 8 \
    --max_completion_length 3500 \
    --max_length 20000 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-6 \
    --save_strategy "steps" \
    --eval_strategy "no" \
    --eval_steps 10000 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --beta 0.04 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 16 \
    --max_grad_norm 1.0 \
    --temperature 0.6 \
    --top_p 0.9 \
    --top_k 50 \
    --repetition_penalty 1 \
    --check_model false \
    --report_to tensorboard wandb \
    --num_iterations 2 \
    --deepspeed zero3 \
    --overlong_filter false \
    --attn_impl flash_attn \
    --epsilon 0.2 \
    --log_completions true \
    --async_generate false \
    --dataloader_drop_last true \
    --offload_optimizer true \
    --offload_model true \
    --padding_free true \
    --gc_collect_after_offload true \
    --logging_dir $tf_logging_dir \
    --save_only_model true 2>&1 | tee $log_file