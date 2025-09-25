ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SCRIPT_PATH="${ROOT_DIR}/scripts"
SRC_PATH="${ROOT_DIR}/src"
DATASET_PATH="${ROOT_DIR}/data/len_diff_chat_subset/"
input_file=${ROOT_DIR}/analysis/batch_prompt.json
output_file=${ROOT_DIR}/analysis/batch_prompt_output.json

VLLM_USE_V1=0 PYTHONPATH=${SRC_PATH} python ${SRC_PATH}/ddj_attribution.py \
  --input_file ${input_file} \
  --output_file ${output_file}