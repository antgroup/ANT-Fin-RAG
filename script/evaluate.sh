

set -o pipefail

set -x

mkdir -p /root/.cache/huggingface/hub
echo 1 > /root/.cache/huggingface/hub/version.txt

nproc_per_node=1
RANDOM_PORT=$[$RANDOM + 20000]


NODE_NAME=`echo $ILOGTAIL_PODNAME | awk -F 'ptjob-' '{print $2}'`
NODE_NAME=${NODE_NAME:-master-0}

export NCCL_DEBUG=DEBUG
export NCCL_IB_GID_INDEX=3
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64
export NCCL_SOCKET_NTHREADS=8
export NCCL_IB_TIMEOUT=22
export NCCL_ASYNC_ERROR_HANDLING=1 
export NCCL_P2P_LEVEL=NVL

mkdir -p $output_dir
log_file=$output_dir/log-inference-$NODE_NAME.txt


#!/bin/bash
dir='/AntFinRAG'
datasetname='BizFin'
model_type='qwen3'
dataset="../benchmark/ANT-Fin-RAG/benchmark/FDD_Hard.json" 
model_dirs=(
    'output_qwen3/',
    'output_llama3/'
    )


for model_dir in "${model_dirs[@]}"; do
    echo "============================================"
    echo "Processing model: $model_dir"
    echo "============================================"
    
    IFS='/' read -ra dir_parts <<< "$model_dir"
    dir_part=${dir_parts[-3]}
    
    result_path="${dir}/predictions/${dir_part}_${datasetname}_fdd_prediction.json"
    score_path="${dir}/scores/${dir_part}_${datasetname}_fdd_prediction.json"
    log_file="${dir}/inference.log"
    cd ..//code/ms-swift-rlfkv/
    # inferencing
    NNODES=${WORLD_SIZE:-1} \
    NODE_RANK=${RANK:-0} \
    MASTER_ADDR=${MASTER_ADDR:-127.0.0.1} \
    MASTER_PORT=${MASTER_PORT:-$RANDOM_PORT} \
    NPROC_PER_NODE=$nproc_per_node \
    CUDA_VISIBLE_DEVICES=1 swift infer \
        --model "$model_dir" \
        --model_type "$model_type" \
        --stream false \
        --infer_backend vllm \
        --val_dataset "$dataset" \
        --result_path "$result_path" \
        --max_new_tokens 2048 2>&1 | tee "$log_file"
    # scoring
    cd ../utils/
    python biz_evaluation.py \
    --input_file "$result_path" \
    --scores_file "$score_path"

    echo "Finished processing: $model_dir"
done

echo "All models processed!"