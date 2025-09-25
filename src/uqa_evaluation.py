import datasets
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer
from llm import LLM
import argparse
import tiktoken
import random
import math
from datetime import datetime, timedelta
import os
from utils import extract_json_from_text, validate_and_parse_json_output
import json

import re
def hard_match(text: str):
    """
    Extract the final answer from unstructured LLM output.
    Handles:
    - answer keys with optional quotes (single or double)
    - values wrapped in single, double, or no quotes
    - multiple choice (A-D) or numeric (with commas/decimals)
    """
    # Normalize common escape patterns
    normalized = text.replace('\\"', '"').replace('\\\"', '"').replace("\\'", "'")

    # Flexible regex patterns to match both 'answer' and "answer", and both single/double-quoted or unquoted values

    # 1. Match multiple choice: A/B/C/D
    pattern_choice = re.compile(
        r"[\"']?answer[\"']?\s*:\s*[\"']?([A-D])[\"']?", re.IGNORECASE)

    # 2. Match number (optional comma and decimal)
    pattern_number = re.compile(
        r"[\"']?answer[\"']?\s*:\s*[\"']?([+-]?[0-9]{1,10}(?:,[0-9]{3})*(?:\.\d+)?|\d+\.\d+|\d+)[\"']?"
    )

    # Try to find multiple-choice style answer
    match = pattern_choice.search(normalized)
    if match:
        return match.group(1).strip()

    # Try to find numeric style answer
    match = pattern_number.search(normalized)
    if match:
        return match.group(1).replace(",", "").strip()

    return ""

urgent_prompts = [
    "I'm in a big hurry right now. Please give me the answer quickly!!!",
    "Time is running out! I need your response immediately!!!",
    "Can you please answer this as fast as possible? It's really urgent!!!",
    "I need your help urgently! Every second counts right now!!!",
    "This can't wait! I need your answer immediately!!!",
    "Please hurry! This issue is blocking my progress!!!",
    "I don't have much time! Please give me the answer right away!!!",
    "It's critical that I get a response now! Please be quick!!!",
    "I'm in a rush and need a quick answer to move forward!!!",
    "I'm counting on your quick reply! It’s really urgent!!!",
]

concise_prompts = [
    "Please keep your response brief and to the point!!!",
    "A short and concise answer would be perfect!!!",
    "Kindly respond in the most concise form!!!",
    "I'd prefer a brief and focused reply!!!",
    "Just a short and clear answer, please!!!",
    "Please make your response as concise as possible!!!",
    "Keep it brief—just a concise reply is all I need!!!",
    "I'm looking for a short and compact response!!!",
    "Please stick to a brief reply format!!!",
    "Make your answer short and neatly expressed, please!!!",
]

dataset_info = {
    "obqa": {
        "task_type": "mc",
    },
    "gsm": {
        "task_type": "math",
    },
    "gpqa": {
        "task_type": "mc",
    },
}

def add_args(parser):
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Name of the model to use for evaluation.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./data/",
        help="Path to the dataset for evaluation.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directory to save the evaluation results.",
    )
    parser.add_argument(
        "--save_dataset",
        action="store_true",
        help="Whether to save the dataset after processing.",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=1,
        help="Number of runs for the evaluation.",
    )
    parser.add_argument(
        "--run_mode",
        type=str,
        default="all",
        choices=["all", "normal", "urgent", "concise"],
        help="Mode to run the evaluation. 'all' for both normal and urgent, 'normal' for normal only, 'urgent' for urgent only, 'concise' for concise only.",
    )
    # return parser.parse_args()


SYS_PROMPT = """
You are a helpful assistant.
"""

NORMAL_MATH_PROMPT = """
Solve the following math problem efficiently and clearly. The last line of your response should be a JSON object: {{"answer": "<NUMBER>"}} where NUMBER is the exact numeric answer to the question.
You can write down your step-by-step reasoning process before providing the answer, but please make sure to give the final answer in the required format.

### Question
{Question}
"""

URGENT_MATH_PROMPT = """
Solve the following math problem efficiently and clearly. The last line of your response should be a JSON object: {{"answer": "<NUMBER>"}} where NUMBER is the exact numeric answer to the question.
You can write down your step-by-step reasoning process before providing the answer, but please make sure to give the final answer in the required format.

### Question
{Question}

{urgent_words}
"""

NORMAL_MC_PROMPT = """
Answer the following multiple choice question. The last line of your response should be a JSON object: {{"answer": "LETTER"}} where LETTER is one of ABCD.
You can write down your step-by-step reasoning process before providing the answer, but please make sure to give the final answer in the required format.

### Question
{Question}

A) {A}
B) {B}
C) {C}
D) {D}
"""


URGENT_MC_PROMPT = """
Answer the following multiple choice question. The last line of your response should be a JSON object: {{"answer": "LETTER"}} where LETTER is one of ABCD.
You can write down your step-by-step reasoning process before providing the answer, but please make sure to give the final answer in the required format.

### Question
{Question}

A) {A}
B) {B}
C) {C}
D) {D}

{urgent_words}
"""

def create_mc_samples(dataset: Dataset, dataset_name: str):
    def create_one_sample(item):
        question = item["question"]
        choices = item["choices"] # list of text choices
        label = item["label"] # 0, 1, 2, 3
        # shuffle the choices, but make sure the label is still correct
        choice_idx = list(range(len(choices)))
        random.shuffle(choice_idx)
        shuffled_choices = [choices[i] for i in choice_idx]
        shuffled_label = choice_idx.index(label)
        # label id to label letter
        label_letter = ["A", "B", "C", "D"][shuffled_label]
        # create the sample
        normal_prompt = NORMAL_MC_PROMPT.format(
            Question=question,
            A=shuffled_choices[0],
            B=shuffled_choices[1],
            C=shuffled_choices[2],
            D=shuffled_choices[3],
        )
        urgent_prompt = URGENT_MC_PROMPT.format(
            Question=question,
            A=shuffled_choices[0],
            B=shuffled_choices[1],
            C=shuffled_choices[2],
            D=shuffled_choices[3],
            urgent_words=random.choice(urgent_prompts),
        )

        concise_prompt = URGENT_MC_PROMPT.format(
            Question=question,
            A=shuffled_choices[0],
            B=shuffled_choices[1],
            C=shuffled_choices[2],
            D=shuffled_choices[3],
            urgent_words=random.choice(concise_prompts),
        )

        return {
            "normal_prompt": normal_prompt,
            "urgent_prompt": urgent_prompt,
            "concise_prompt": concise_prompt,
            "label": label_letter,
        }
    dataset = dataset.map(
        create_one_sample,
        remove_columns=dataset.column_names,
    )
    normal_dataset = dataset.map(
        lambda x: {
            "messages": [
                {"role": "system", "content": SYS_PROMPT.strip()},
                {"role": "user", "content": x["normal_prompt"].strip()},
            ],
            "answer": x["label"],
            "dataset_name": dataset_name,
        },
        remove_columns=dataset.column_names,
    )
    urgent_dataset = dataset.map(
        lambda x: {
            "messages": [
                {"role": "system", "content": SYS_PROMPT.strip()},
                {"role": "user", "content": x["urgent_prompt"].strip()},
            ],
            "answer": x["label"],
            "dataset_name": dataset_name,
        },
        remove_columns=dataset.column_names,
    )
    concise_dataset = dataset.map(
        lambda x: {
            "messages": [
                {"role": "system", "content": SYS_PROMPT.strip()},
                {"role": "user", "content": x["concise_prompt"].strip()},
            ],
            "answer": x["label"],
            "dataset_name": dataset_name,
        },
        remove_columns=dataset.column_names,
    )
    return normal_dataset, urgent_dataset, concise_dataset


def create_math_samples(dataset: Dataset, dataset_name: str):
    def create_one_sample(item):
        question = item["question"]
        label = item["numerical_answer"] 
        normal_prompt = NORMAL_MATH_PROMPT.format(
            Question=question,
        )
        urgent_prompt = URGENT_MATH_PROMPT.format(
            Question=question,
            urgent_words=random.choice(urgent_prompts),
        )
        concise_prompt = URGENT_MATH_PROMPT.format(
            Question=question,
            urgent_words=random.choice(concise_prompts),
        )
        return {
            "normal_prompt": normal_prompt,
            "urgent_prompt": urgent_prompt,
            "concise_prompt": concise_prompt,
            "label": label,
        }
    dataset = dataset.map(
        create_one_sample,
        remove_columns=dataset.column_names,
    )
    normal_dataset = dataset.map(
        lambda x: {
            "messages": [
                {"role": "system", "content": SYS_PROMPT.strip()},
                {"role": "user", "content": x["normal_prompt"].strip()},
            ],
            "answer": x["label"],
            "dataset_name": dataset_name,
        },
        remove_columns=dataset.column_names,
    )
    urgent_dataset = dataset.map(
        lambda x: {
            "messages": [
                {"role": "system", "content": SYS_PROMPT.strip()},
                {"role": "user", "content": x["urgent_prompt"].strip()},
            ],
            "answer": x["label"],
            "dataset_name": dataset_name,
        },
        remove_columns=dataset.column_names,
    )
    concise_dataset = dataset.map(
        lambda x: {
            "messages": [
                {"role": "system", "content": SYS_PROMPT.strip()},
                {"role": "user", "content": x["concise_prompt"].strip()},
            ],
            "answer": x["label"],
            "dataset_name": dataset_name,
        },
        remove_columns=dataset.column_names,
    )
    return normal_dataset, urgent_dataset, concise_dataset

def evaluate_dataset(
    llm: LLM,
    dataset: Dataset,
    runs: int = 1,
):
    final_results = {"runs": []}
    for n in range(runs):
        print(f"Run {n + 1}/{runs}...")
        messages = dataset["messages"]
        labels = dataset["answer"]
        dataset_names = dataset["dataset_name"]
        results = []

        responses = llm.generate(messages)['responses']
        for i, response in enumerate(responses):
            try:
                if type(response) == dict:
                    solution = response.get("solution", "")
                    reasoning = response.get("reasoning", "")
                    response = f"<think>{reasoning}</think><solution>{solution}</solution>"
                else:
                    solution = response
                    reasoning = ""
                    
                parsed_response = validate_and_parse_json_output(response)
                if parsed_response is None:
                    answer = hard_match(response)
                    if answer == "":
                        print(f"Invalid JSON output for sample {i}: {responses[i]}")
                else:
                    answer = str(parsed_response.get("answer", "")).replace(",", "")
                result = {
                    "run": n,
                    "dataset_name": dataset_names[i],
                    "messages": messages[i],
                    "response": response,
                    "parsed_response": parsed_response,
                    "answer": answer,
                    "label": labels[i],
                    "correct": answer == labels[i],
                }
                results.append(result)
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
        # compute accuracy
        correct_count = sum(1 for result in results if result["correct"])
        total_count = len(results)
        accuracy = correct_count / total_count if total_count > 0 else 0
        run_results = {
            "accuracy": accuracy,
            "total_count": total_count,
            "correct_count": correct_count,
            "results": results,
        }
        final_results["runs"].append(run_results)
    return final_results

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Time-aware QA tasks."
    )
    add_args(parser)
    LLM.add_arguments(parser)
    args = parser.parse_args()

    model_name = args.llm_in_use
    output_dir = args.output_dir
    # Load the dataset
    all_datasets = {}
    for dataset_name, detail in dataset_info.items():
        dataset_path = f"{args.dataset_path}/{dataset_name}"
        if not os.path.exists(dataset_path):
            print(f"Dataset {dataset_name} not found at {dataset_path}.")
            continue
        dataset = load_from_disk(dataset_path)
        all_datasets[dataset_name] = dataset
        print(f"Loaded dataset from {dataset_path}.")
    # Check output dir and see if all datasets exist
    output_dir = f"{output_dir}/{model_name.replace('/', '_')}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create the dataset
    data_creator = {
        "math": create_math_samples,
        "mc": create_mc_samples,
    }

    all_normal_datasets = []
    all_urgent_datasets = []
    all_concise_datasets = []

    if os.path.exists(f"{output_dir}/normal_dataset") and os.path.exists(f"{output_dir}/urgent_dataset") and os.path.exists(f"{output_dir}/concise_dataset"):
        print(f"datasets already exists at {output_dir}/.")
        normal_dataset = load_from_disk(f"{output_dir}/normal_dataset")
        urgent_dataset = load_from_disk(f"{output_dir}/urgent_dataset")
        concise_dataset = load_from_disk(f"{output_dir}/concise_dataset")
    else:
        for dataset_name, detail in dataset_info.items():
            dataset = all_datasets[dataset_name]
            task_type = detail["task_type"]
            if task_type not in data_creator:
                print(f"Task type {task_type} not supported for dataset {dataset_name}.")
                continue
            data_creator_func = data_creator[task_type]
            normal_dataset, urgent_dataset, concise_dataset = data_creator_func(dataset, dataset_name)
            all_normal_datasets.append(normal_dataset)
            all_urgent_datasets.append(urgent_dataset)
            all_concise_datasets.append(concise_dataset)
        # Concatenate all datasets
        normal_dataset = datasets.concatenate_datasets(all_normal_datasets)
        urgent_dataset = datasets.concatenate_datasets(all_urgent_datasets)
        concise_dataset = datasets.concatenate_datasets(all_concise_datasets)
        if args.save_dataset:
            normal_dataset.save_to_disk(f"{output_dir}/normal_dataset")
            urgent_dataset.save_to_disk(f"{output_dir}/urgent_dataset")
            concise_dataset.save_to_disk(f"{output_dir}/concise_dataset")
    # Load the model
    llm = LLM(args)
    llm.initialize()
    # Evaluate the model on type 1 dataset
    if args.run_mode == "normal" or args.run_mode == "all":
        print("Evaluating Normal dataset...")
        normal_results = evaluate_dataset(llm, normal_dataset, runs=args.num_runs)
        # Save the results
        with open(f"{output_dir}/normal_results.json", "w") as f:
            json.dump(normal_results, f, indent=4)
        avg_accuracy = sum(run["accuracy"] for run in normal_results["runs"]) / len(normal_results["runs"])
        print(f"Normal dataset evaluation results: {avg_accuracy:.2%} accuracy")
    
    if args.run_mode == "urgent" or args.run_mode == "all":
        # Evaluate the model on urgent dataset
        print("Evaluating Urgent dataset...")
        urgent_results = evaluate_dataset(llm, urgent_dataset, runs=args.num_runs)
        # Save the results
        with open(f"{output_dir}/urgent_results.json", "w") as f:
            json.dump(urgent_results, f, indent=4)
        # Compute the average accuracy
        avg_accuracy = sum(run["accuracy"] for run in urgent_results["runs"]) / len(urgent_results["runs"])
        print(f"Urgent dataset evaluation results: {avg_accuracy:.2%} accuracy")
    
    if args.run_mode == "concise" or args.run_mode == "all":
        # Evaluate the model on concise dataset
        print("Evaluating Concise dataset...")
        concise_results = evaluate_dataset(llm, concise_dataset, runs=args.num_runs)
        # Save the results
        with open(f"{output_dir}/concise_results.json", "w") as f:
            json.dump(concise_results, f, indent=4)
        # Compute the average accuracy
        avg_accuracy = sum(run["accuracy"] for run in concise_results["runs"]) / len(concise_results["runs"])
        print(f"Concise dataset evaluation results: {avg_accuracy:.2%} accuracy")
    print("Evaluation completed.")
    llm.unload()


if __name__ == "__main__":
    main()