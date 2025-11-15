# !/bin/bash


model=$1
model_short_name=$2
think_time_consume=$3
map_size="(8,8)"
wall_density=0.15
initial_remaining_time=300
max_steps=20
max_runs=100
action_time_consume="{'get_state':1,'move':1,'think':${think_time_consume},'detect':1,'invalid':1}"
time_measurement_mode="token"



task_type="treasure"
output_dir="./bombrush_exp/${model_short_name}_${task_type}/"
log_name="${model_short_name}_${task_type}.log"
mkdir -p ${output_dir}
PYTHONPATH=./src/ python src/evaluation.py --inference_mode api \
  --api_key None \
  --base_url http://127.0.0.1:9876/v1/ \
  --llm_in_use ${model} \
  --grid_size ${map_size} \
  --wall_density ${wall_density} \
  --initial_remaining_time ${initial_remaining_time} \
  --action_time_consume ${action_time_consume} \
  --time_measurement_mode ${time_measurement_mode} \
  --max_steps ${max_steps} \
  --task_type ${task_type} \
  --disable_time \
  --disable_bomb_move \
  --disable_detect \
  --max_runs ${max_runs} \
  --num_workers 2 \
  --output_dir ${output_dir} \
  --self_serve_vllm \
  --max_tokens 24576 \
  --debug | tee ${output_dir}/${log_name}


task_type="static_bomb"
output_dir="./bombrush_exp/${model_short_name}_${task_type}/"
log_name="${model_short_name}_${task_type}.log"
mkdir -p ${output_dir}
PYTHONPATH=./src/ python src/evaluation.py --inference_mode api \
  --api_key None \
  --base_url http://127.0.0.1:9876/v1/ \
  --llm_in_use ${model} \
  --grid_size ${map_size} \
  --wall_density ${wall_density} \
  --initial_remaining_time ${initial_remaining_time} \
  --action_time_consume ${action_time_consume} \
  --time_measurement_mode ${time_measurement_mode} \
  --max_steps ${max_steps} \
  --task_type ${task_type} \
  --disable_bomb_move \
  --disable_detect \
  --max_runs ${max_runs} \
  --num_workers 2 \
  --output_dir ${output_dir} \
  --self_serve_vllm \
  --max_tokens 24576 \
  --debug | tee ${output_dir}/${log_name}


max_steps=30

task_type="moving_bomb_detect"
output_dir="./bombrush_exp/${model_short_name}_${task_type}/"
log_name="${model_short_name}_${task_type}.log"
mkdir -p ${output_dir}
PYTHONPATH=./src/ python src/evaluation.py --inference_mode api \
  --api_key None \
  --base_url http://127.0.0.1:9876/v1/ \
  --llm_in_use ${model} \
  --grid_size ${map_size} \
  --wall_density ${wall_density} \
  --initial_remaining_time ${initial_remaining_time} \
  --action_time_consume ${action_time_consume} \
  --time_measurement_mode ${time_measurement_mode} \
  --max_steps ${max_steps} \
  --task_type ${task_type} \
  --bomb_move_ratio 3 \
  --max_runs ${max_runs} \
  --num_workers 2 \
  --output_dir ${output_dir} \
  --self_serve_vllm \
  --max_tokens 24576 \
  --debug | tee ${output_dir}/${log_name}