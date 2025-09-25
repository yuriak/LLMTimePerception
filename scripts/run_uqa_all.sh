ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SCRIPT_PATH="${ROOT_DIR}/scripts"
SRC_PATH="${ROOT_DIR}/src"
OUTPUT_DIR="${ROOT_DIR}/uqa_result/"

models=(
  "meta-llama/Llama-3.1-8B-Instruct"
  "Qwen/Qwen2.5-7B-Instruct"
  "meta-llama/Llama-3.3-70B-Instruct"
  "Qwen/Qwen2.5-72B-Instruct"
)

job_script=$1
uqa_script="${SCRIPT_PATH}/run_uqa.sh"
run_mode="concise"
num_runs=1

for model in "${models[@]}"; do
    model_short_name=$(echo $model | tr '/' '_')
    sbatch -J "uqa_${model_short_name}" ${job_script} $uqa_script $model $OUTPUT_DIR $num_runs $run_mode
    # ${uqa_script} ${model} ${OUTPUT_DIR} ${num_runs} ${run_mode}
done


models=(
  "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
  "Qwen/QwQ-32B"
)

uqa_script="${SCRIPT_PATH}/run_uqa_lrm.sh"

for model in "${models[@]}"; do
    model_short_name=$(echo $model | tr '/' '_')
    sbatch -J "uqa_${model_short_name}" ${job_script} $uqa_script $model $OUTPUT_DIR $num_runs $run_mode
    # ${uqa_script} ${model} ${OUTPUT_DIR} ${num_runs} ${run_mode}
done