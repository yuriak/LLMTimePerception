# 🕒 **Discrete Minds in a Continuous World: Do Language Models Know Time Passes?**

*A clean, runnable reference implementation accompanying the [EMNLP 2025 paper](https://aclanthology.org/2025.findings-emnlp.1016/).*
This repository contains all code and datasets used in our experiments on **LLM time perception**, including dialogue duration judgment (DDJ), urgency-aware question answering (UQA), and BombRush time-pressured decision making.

---

## 🧠 Overview

Modern large language models (LLMs) show strong reasoning and world-knowledge capabilities—but can they **perceive time** as it passes?
This project investigates that question through the **Token-Time Hypothesis**, which posits that LLMs may treat token counts as discrete proxies for real-world, continuous time. We explore whether this implicit mapping influences how they interpret, reason, and act under temporal constraints.

### Core Research Questions

1. **Dialogue Duration Judgment (DDJ): Do LLMs associate token length with elapsed time?**
   In this experiment, models are given two multi-turn conversations and asked to judge which one took longer to generate.
   We test this under multiple cue conditions—purely textual, explicit timestamps, and intentionally misleading signals—to see whether models rely on text length, timing information, or a deeper mapping between token count and wall-clock time.
   The DDJ task validates the *Token-Time Hypothesis* by showing that most models can infer duration from text length and that large reasoning models (LRMs) remain robust even under contradictory temporal cues.

2. **Urgency-Aware Question Answering (UQA): Can LLMs demonstrate “temporal empathy”?**
   Temporal empathy means adapting behavior to user time constraints.
   In UQA, we compare model accuracy and output length between *normal* and *urgent* prompts (e.g., “Please answer quickly!!!”).
   Models consistently shorten responses under urgency while maintaining or improving accuracy—especially on reasoning benchmarks—suggesting that they internalize the relation between shorter token sequences and reduced response time.

3. **BombRush: How do LLMs plan and reason when time literally runs out?**
   BombRush extends the study from static tasks to a dynamic, interactive environment.
   An LLM agent navigates a grid world to locate a hidden bomb before it “explodes,” with every reasoning token consuming simulated time.
   The agent must balance thoughtfulness and action, dynamically adjusting reasoning depth as time diminishes.
   This experiment reveals that reasoning verbosity decreases as temporal pressure increases, demonstrating adaptive time-aware decision making.

Together, these experiments offer the first systematic evidence that LLMs possess a primitive but measurable awareness of temporal progression, bridging the gap between discrete linguistic processing and continuous real-world time.

---

## ⚙️ Installation

**Requirements**

* Python 3.10 +
* CUDA GPU recommended for self-hosted inference

```bash
pip install vllm transformers accelerate torch datasets tiktoken pydantic tqdm httpx openai azure-ai-inference ray[default]
```

Notes:

* For GPU builds of `torch`, follow official PyTorch installation instructions.
* `openai` or `azure-ai-inference` clients can communicate with any OpenAI-compatible API.
* `vLLM` can self-host an OpenAI-compatible HTTP endpoint for local experiments.

---

## 🔌 LLM Backend Options

`src/llm.py` supports:

* `api` — Connect to any OpenAI-compatible HTTP endpoint (including vLLM)
* `api_self_hosted` — Automatically start a local vLLM server
* `azure` — Use Azure OpenAI (`--base_url` must include `api-version=...`)
* `vllm` — Direct, in-process GPU inference

Common arguments:

```
--llm_in_use meta-llama/Llama-3.3-70B-Instruct
--base_url http://localhost:9876/v1/
--api_key None
--max_batch_size 32 --max_tokens 16384
```

---

## 🚀 Quick Start

### Option A — Manual vLLM Server

```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --host 0.0.0.0 --port 9876 \
  --dtype bfloat16 --tensor-parallel-size 2 \
  --max-model-len 16384 --enable-prefix-caching \
  --distributed-executor-backend ray
```

Then run with
`--inference_mode api  --base_url http://localhost:9876/v1/  --api_key None`.

### Option B — Automatic Self-Hosting

Use `--inference_mode api_self_hosted`; the runner will start/stop vLLM automatically.

---

## 🧪 Running Experiments

All commands assume repository root; add `PYTHONPATH=src` when invoking.

### 1️⃣ Dialogue Duration Judgment (DDJ)

```bash
PYTHONPATH=src python src/ddj_evaluation.py \
  --inference_mode api_self_hosted \
  --base_url http://localhost:9876/v1/ \
  --api_key None \
  --llm_in_use meta-llama/Llama-3.3-70B-Instruct \
  --dataset_path ./data/len_diff_chat_subset \
  --output_dir ./ddj_result \
  --num_runs 1 \
  --save_dataset
```

To evaluate rationale categories (e.g., *text_length*, *time*, *semantic*):

```bash
PYTHONPATH=src python src/ddj_attribution.py \
  --input_file ./ddj_result/<MODEL>/prompts.json \
  --output_file ./ddj_result/<MODEL>/ddj_attribution.json
```

---

### 2️⃣ Urgency-Aware QA (UQA)

```bash
PYTHONPATH=src python src/uqa_evaluation.py \
  --inference_mode api_self_hosted \
  --base_url http://localhost:9876/v1/ \
  --api_key None \
  --llm_in_use meta-llama/Llama-3.3-70B-Instruct \
  --dataset_path ./data \
  --output_dir ./uqa_result \
  --num_runs 1 \
  --run_mode all \
  --save_dataset \
  --max_batch_size 32 \
  --max_tokens 16384
```

To compare *normal* vs *urgent* responses:

```bash
PYTHONPATH=src python src/uqa_attribution.py \
  --input_file_a ./uqa_result/<MODEL>/normal_results.json \
  --input_file_b ./uqa_result/<MODEL>/urgent_results.json \
  --output_file  ./uqa_result/<MODEL>/uqa_attribution.json
```

---

### 3️⃣ BombRush (Time-Aware Decision Making)

```bash
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
```

Available task variants (see `GridWorldEnv.add_arguments` inside `bombrush_grid_world.py`):
`treasure`, `static_bomb`, `static_bomb_hint`, `static_bomb_hint_urge`, `static_bomb_urge2_no_hint`, `moving_bomb`, `moving_bomb_detect`.
Each reasoning token consumes simulated time; results (trajectories, metrics) are saved as `.pkl`.

---

## 📊 Reproducing Results and Figures

1. **DDJ** → Run all cue variants (text-only, timestamp, misleading). Aggregate accuracy and rationale distributions.
2. **UQA** → Compute accuracy and token-length deltas between normal and urgent conditions.
3. **BombRush** → Aggregate success rates, reasoning length vs remaining time, and step-wise adaptation curves.

All hyperparameters and example model lists appear in `scripts/run_*_all.sh`.

---

## 🧩 Key Findings

* **Token-Time Mapping:** LLMs associate output length with elapsed generation time.
* **Temporal Empathy:** Under urgency, models become concise without losing accuracy.
* **Adaptive Reasoning:** In BombRush, reasoning verbosity decreases as time dwindles.
* **Model Scale Matters:** Large reasoning models (e.g., DeepSeek-R1, QwQ-32B) handle conflicting temporal cues far better than smaller LLMs.

These observations support the view that LLMs exhibit an emergent, quantifiable **sense of time passage**.

---

## 🧾 Citation

```bibtex
@inproceedings{wang-etal-2025-discrete,
    title = "Discrete Minds in a Continuous World: Do Language Models Know Time Passes?",
    author = "Wang, Minghan  and
      Bai, Ye  and
      Vu, Thuy-Trang  and
      Shareghi, Ehsan  and
      Haffari, Gholamreza",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2025",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-emnlp.1016/",
    pages = "18703--18729",
    ISBN = "979-8-89176-335-7",
}
```