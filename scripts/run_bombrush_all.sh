# !/bin/bash

job_script="submit_job_a100.sh"

# Define models, short names, and think times in arrays
models=(
  "meta-llama/Llama-3.3-70B-Instruct"
  "Qwen/Qwen2.5-72B-Instruct"
  "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
  "Qwen/QwQ-32B"
)

model_short_names=(
  "l370"
  "q257"
  "dl370"
  "qwq"
)

think_time_consumes=(
  0.1
  0.1
  0.01
  0.01
)

# Loop through the arrays and submit jobs
for i in "${!models[@]}"; do
  model="${models[$i]}"
  model_short_name="${model_short_names[$i]}"
  think_time_consume="${think_time_consumes[$i]}"

  sbatch -J "${model_short_name}" "${job_script}" run_bombrush.sh "${model}" "${model_short_name}" "${think_time_consume}"
done