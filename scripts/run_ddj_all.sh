ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SCRIPT_PATH="${ROOT_DIR}/scripts"
SRC_PATH="${ROOT_DIR}/src"
OUTPUT_DIR="${ROOT_DIR}/tpqa_result/"

models=(
  "meta-llama/Llama-3.3-70B-Instruct"
  "Qwen/Qwen2.5-72B-Instruct"
  "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
  "Qwen/QwQ-32B"
  "meta-llama/Llama-3.1-8B-Instruct"
  "Qwen/Qwen2.5-7B-Instruct"
)

job_script_name=$1
job_script=$1
tpqa_script="${SCRIPT_PATH}/run_tpqa.sh"

num_runs=5

# run all experiments with slurm, modify this to run on local machine
for model in "${models[@]}"; do
    model_short_name=$(echo $model | tr '/' '_')
    sbatch -J "tpqa_${model_short_name}" ${job_script} ${tpqa_script} ${model} $OUTPUT_DIR ${num_runs}
    # ${tpqa_script} ${model} ${OUTPUT_DIR} ${num_runs}
done