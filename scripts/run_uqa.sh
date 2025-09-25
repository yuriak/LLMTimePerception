ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SCRIPT_PATH="${ROOT_DIR}/scripts"
SRC_PATH="${ROOT_DIR}/src"
DATASET_ROOT="${ROOT_DIR}/data/"

llm=$1
output_root=$2
num_runs=$3
run_mode=$4

VLLM_USE_V1=0 PYTHONPATH=$SRC_PATH python ${SRC_PATH}/uqa_evaluation.py --inference_mode api_self_hosted \
    --base_url http://localhost:9876/v1/ \
    --api_key None \
    --llm_in_use $llm \
    --dataset_path ${DATASET_ROOT} \
    --num_runs ${num_runs} \
    --output_dir ${output_root} \
    --max_batch_size 32 \
    --save_dataset \
    --run_mode $run_mode \
    --max_tokens 16384