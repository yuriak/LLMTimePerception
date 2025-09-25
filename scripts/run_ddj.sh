ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SCRIPT_PATH="${ROOT_DIR}/scripts"
SRC_PATH="${ROOT_DIR}/src"
DATASET_PATH="${ROOT_DIR}/data/len_diff_chat_subset/"
model=$1
output_dir=$2
num_runs=$3

VLLM_USE_V1=0 PYTHONPATH=${SRC_PATH} python ${SRC_PATH}/tpqa_evaluation.py --inference_mode api_self_hosted \
  --base_url http://localhost:9876/v1/ \
  --api_key None \
  --llm_in_use ${model} \
  --dataset_path ${DATASET_PATH} \
  --output_dir  ${output_dir} \
  --num_runs ${num_runs} \
  --max_batch_size 16 \
  --save_dataset \
  --max_tokens 16384