import sys
import json
import argparse
import numpy as np
import glob
import pandas as pd

sys_prompt = """
You are a classification agent. Your task is to classify the reasoning provided by an LLM into one of the following four categories, based on **how** the LLM determines which response took longer time to generate:

### Allowed categories (return only one of the category names):
- `time`
- `text_length`
- `semantic`
- `other`

### Classification Rules:

1. **`time`**:  
   The reason explicitly involves **timing information** — such as start time, end time, duration (e.g., “1 minute and 45 seconds”), timestamps, or calculations of elapsed time.  
   If the decision is made **primarily or solely based on these time-based values**, without switching judgment due to other factors, classify it as `time`.

2. **`text_length`**:  
   The reason makes a judgment based on the **length of the text**, such as token count, number of words, number of sentences, or how long the generated response is.  
   This includes explicitly mentioning phrases like “Response A is longer,” “has more tokens,” or “took more space to explain”, etc.

3. **`semantic`**:  
   The reason does **not mention time or length difference** at all, but solely relies on **semantic or cognitive complexity** — such as the depth of explanation, difficulty of the topic, use of logic or math, or other indicators of **conceptual effort**.

4. **`other`**:  
   Use this category if the reasoning doesn’t clearly match any of the above — for example, if the model relies on **irrelevant metadata**, contradictory logic, unclear rationale, or vague comparison that doesn’t fit well into the previous categories.

Do **not** include any explanation or justification in your response.
"""

user_prompt = """

Here is the explanation to classify:
```
{reason}
```

"""

all_models = [
    'Llama-3.1-8B-Instruct',
    'Qwen2.5-7B-Instruct',
    'Llama-3.3-70B-Instruct',
    'Qwen2.5-72B-Instruct',
    'DeepSeek-R1-Distill-Llama-70B',
    'QwQ-32B'
]

all_tasks = [
    'type1_dataset', 
    'type1_easy_dataset',
    'type1_very_easy_dataset', 
    'type2_dataset',
    'type2_misleading_dataset',
    'type2_misleading_with_token_dataset',
]

sample_model = all_models[0]
sample_task = all_tasks[0]

def get_raw_response(text):
    return text.replace("<think>", "").replace("</think>","").replace("<solution>", "").replace("</solution>","")
def get_reason(result):
    if result['parsed_response'] is not None and "reason" in result['parsed_response']:
        return result['parsed_response']['reason']
    return get_raw_response(result['response'])
def promtpify(reason):
    if len(reason) > 10000:
        reason = reason[-10000:]
    return [{
        "role":"system",
        "content": sys_prompt.strip()
    }, {
        "role": "user",
        "content": user_prompt.format(reason=reason).strip()
    }]

all_results = glob.glob("../uqa_result_final_merge/*/*.json")
final_results = {}
for x in all_results:
    model_name, dataset_name = x.split("/")[-2], x.split("/")[-1]
    model_name = model_name.split("_")[-1]
    dataset_name = dataset_name.replace("_results.json","")
    if model_name not in final_results:
        final_results[model_name] = {}
    if dataset_name not in final_results[model_name]:
        final_results[model_name][dataset_name] = json.load(open(x))

all_prompts= []
for sample_model in all_models:
    for sample_task in all_tasks:
        for run in final_results[sample_model][sample_task]['runs']:
            # print(f"Running {sample_model} {sample_task}")
            results = run['results']
            reason_prompt = [ promtpify(get_reason(x)) for x in results]
            all_prompts.extend(reason_prompt)
            # responses = llm.generate(reason_prompt, **eb_args)
            # attributions = list(map(lambda x: x['solution'], responses['responses']))
            # assert len(attributions) == len(reason_prompt[0])
            # for i, result in enumerate(run['results']):
            #     run['results'][i]['attribute'] = attributions[i]

# json.dump(all_prompts, open("./batch_prompt.json", 'w'))

parser = argparse.ArgumentParser(description="Run batch attribution")
parser.add_argument(
    "--input_file",
    type=str,
    default="./batch_prompt.json",
    help="Path to the input file containing prompts",
)
parser.add_argument(
    "--output_file",
    type=str,
    default="./batch_response.json",
    help="Path to the output file to save responses",
)

args = parser.parse_args()
input_file = args.input_file
output_file = args.output_file

eb_args = {
    "extra_body": {"guided_choice": ["time", "text_length", "semantic", "other"]}
}

all_prompt = json.load(open(input_file, "r"))

from llm import LLM

from argparse import Namespace

llm = LLM(
    Namespace(
        inference_mode="api_self_hosted",
        api_key="None",
        base_url="http://localhost:9876/v1/",
        llm_in_use="meta-llama/Llama-3.3-70B-Instruct",
        fast_mode=False,
        max_retry=3,
        max_tokens=16384,
        num_workers=1,
        max_batch_size=16,
    )
)

llm.initialize()

responses = llm.generate(all_prompt, **eb_args)

with open(output_file, "w") as f:
    json.dump(responses, f, indent=4)
print(f"Responses saved to {output_file}")

responses = json.load(open("./batch_prompt_output.json"))
attributions = list(map(lambda x:x['solution'], responses['responses']))
for sample_model in all_models:
    for sample_task in all_tasks:
        for run in final_results[sample_model][sample_task]['runs']:
            # print(f"Running {sample_model} {sample_task}")
            results = run['results']
            # responses = llm.generate(reason_prompt, **eb_args)
            # attributions = list(map(lambda x: x['solution'], responses['responses']))
            # assert len(attributions) == len(reason_prompt[0])
            for i, result in enumerate(run['results']):
                run['results'][i]['attribute'] = attributions.pop(0)
rows = []
for sample_model in all_models:
    for sample_task in all_tasks:
        attr_result = pd.concat([pd.DataFrame(final_results[sample_model][sample_task]['runs'][i]['results'])['attribute'].value_counts() for i in range(5)],axis=1).fillna(0).mean(1)
        rows.append({"model": sample_model, "task": sample_task, **attr_result.to_dict()})

all_attrs = ["time", "text_length", "semantic", "other"]

rows = []
for sample_model in all_models:
    for sample_task in all_tasks:
        correct_ratio = {}
        for attr in all_attrs:
            attr_correct = 0
            attr_incorrect = 0
            attr_count = 0
            for run in final_results[sample_model][sample_task]['runs']:
                for result in run['results']:
                    if result['attribute'] == attr:
                        attr_count +=1
                        if result['correct']:
                            attr_correct +=1
                        else:
                            attr_incorrect +=1
            num_runs = len(final_results[sample_model][sample_task]['runs'])
            attr_correct = attr_correct / num_runs
            attr_incorrect = attr_incorrect / num_runs
            attr_count = attr_count / num_runs
            # correct_ratio[attr] = (attr_correct / attr_count if attr_count > 0 else 0.0 ) * 100
            correct_ratio[attr] = attr_correct
            
        rows.append({"model": sample_model, "task": sample_task, **correct_ratio})
        