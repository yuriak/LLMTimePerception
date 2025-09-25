export RAY_TMPDIR=${HOME}/workspace/ray_tmp
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SCRIPT_PATH="${ROOT_DIR}/scripts"
SRC_PATH="${ROOT_DIR}/src"
DATASET_PATH="${ROOT_DIR}/data/len_diff_chat_subset/"
# modify the input files to run the attribution on different models
input_file_a=${ROOT_DIR}/analysis/uqa_lrm_last_result/Qwen_Qwen2.5-72B-Instruct/normal_results.json
input_file_b=${ROOT_DIR}/analysis/uqa_lrm_last_result/Qwen_Qwen2.5-72B-Instruct/urgent_results.json
output_file=${ROOT_DIR}/analysis/uqa_attribution/Qwen_Qwen2.5-72B-Instruct/attribution_results.json
mkdir -p $(dirname ${output_file})

VLLM_USE_V1=0 PYTHONPATH=${SRC_PATH} python ${SRC_PATH}/uqa_attribution.py \
  --input_file_a ${input_file_a} \
  --input_file_b ${input_file_b} \
  --output_file ${output_file}