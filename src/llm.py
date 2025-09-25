import vllm
import gc
from vllm.sampling_params import GuidedDecodingParams, SamplingParams
from vllm.distributed.parallel_state import (
    destroy_model_parallel,
    destroy_distributed_environment,
)
from openai import OpenAI, AzureOpenAI
import torch
from transformers import AutoTokenizer, GenerationConfig
from pydantic import BaseModel
import tqdm
from utils import validate_and_parse_json_output, post_process_output
import logging
import json
import argparse
import os
import contextlib
import tiktoken
from multiprocessing import Pool
import subprocess
import requests
import time
import signal
from openai import AsyncOpenAI
import asyncio
from tqdm.asyncio import tqdm_asyncio
from asyncio import Semaphore
import httpx
from pathlib import Path
import uuid

logger = logging.getLogger(__name__)
long_timeout_client = httpx.Client(timeout=600)
long_timeout_async_client = httpx.AsyncClient(timeout=600)


def start_vllm(llm_in_use, max_tokens=16384):
    env = os.environ.copy()
    env["VLLM_USE_V1"] = "0"
    env["RAY_TMPDIR"] = f"/tmp/ray_mh/{uuid.uuid4()}"
    reasoning_model = (
        True if any([u in llm_in_use for u in ["deepseek", "QWQ"]]) else False
    )
    model_arg = f"--model {llm_in_use}"
    if reasoning_model:
        model_arg += " --reasoning-parser deepseek_r1 --enable-reasoning"

    vllm_cmd = f"""
    python -m vllm.entrypoints.openai.api_server \
    {model_arg} \
    --dtype bfloat16 \
    --api-key None \
    --tensor-parallel-size 2 \
    --host 0.0.0.0 \
    --port 9876 \
    --max-model-len {max_tokens} \
    --distributed-executor-backend ray \
    --enable-chunked-prefill \
    --gpu_memory_utilization 0.95 \
    --disable-uvicorn-access-log \
    --disable-log-stats \
    --disable-log-requests \
    --enable-prefix-caching
    """

    process = subprocess.Popen(
        vllm_cmd.strip().split(),
        # stdout=subprocess.PIPE,
        # stderr=subprocess.STDOUT,
        env=env,
        preexec_fn=os.setsid,  # 为了后续使用pgid杀掉整个进程组
    )
    logger.info(f"vLLM server started with PID {process.pid}.")
    return process


def wait_for_vllm_ready(port):
    logging.info("waiting for vLLM server to be ready...")
    url = f"http://localhost:{port}/v1/models"
    for _ in range(60 * 15):  # 15 minutes timeout
        try:
            res = requests.get(url, timeout=1)
            if res.status_code == 200:
                logger.info("vLLM server is ready.")
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(20)
    logger.info("vLLM server is not ready after 15 minutes.")
    return False


def stop_vllm(process):
    logger.info("Stopping vLLM server...")
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        process.wait()
        logger.info("vLLM server stopped.")
    except Exception as e:
        logger.info(f"Error stopping vLLM server: {e}")


class LLM:
    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        parser.add_argument(
            "--inference_mode",
            type=str,
            default="api",
            choices=["api", "vllm", "azure", "api_self_hosted", "api_async"],
            help="Inference mode to use: 'api' for OpenAI API, 'vllm' for VLLM.",
        )
        parser.add_argument(
            "--api_key",
            type=str,
            default=None,
            help="API key for OpenAI API. Required if inference_mode is 'api'.",
        )
        parser.add_argument(
            "--base_url",
            type=str,
            default=None,
            help="Base URL for OpenAI API. Required if inference_mode is 'api'.",
        )
        parser.add_argument(
            "--llm_in_use",
            type=str,
            default="meta-llama/Llama-3.3-70B-Instruct",
            help="Model name to use for inference.",
        )
        parser.add_argument(
            "--fast_mode",
            action="store_true",
            default=False,
            help="Use fast mode for inference. First use unguided decoding, then guided decoding if needed.",
        )
        parser.add_argument(
            "--max_retry",
            type=int,
            default=3,
            help="Maximum number of retries for API requests.",
        )
        parser.add_argument(
            "--max_tokens",
            type=int,
            default=16384,
            help="Maximum number of tokens to generate.",
        )
        parser.add_argument(
            "--max_batch_size",
            type=int,
            default=256,
            help="Maximum batch size for inference.",
        )

    def __init__(self, args):
        self.args = args
        self.inference_mode = args.inference_mode
        self.fast_mode = args.fast_mode
        self.max_retry = args.max_retry
        self.max_tokens = args.max_tokens
        self.max_batch_size = args.max_batch_size
        self.chat_history = []
        debug = getattr(args, "debug", False)
        if debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
        # We allow two types of inference modes: 'api' and 'vllm'

    def reset(self):
        self.chat_history = []

    def initialize(self):
        if self.inference_mode == "api" or self.inference_mode == "api_self_hosted":
            self.api_key = self.args.api_key
            self.base_url = self.args.base_url
            self.model = self.args.llm_in_use
            self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url, http_client=long_timeout_async_client)
            if self.inference_mode == "api_self_hosted":
                self.process = start_vllm(self.model, max_tokens=self.max_tokens)
                if wait_for_vllm_ready(9876):
                    logger.info("vLLM server is ready.")
        elif self.inference_mode == "azure":
            self.api_key = self.args.api_key
            self.base_url = self.args.base_url
            self.model = self.args.llm_in_use
            api_version = self.base_url.split("api-version=")[-1]
            self.client = AzureOpenAI(
                api_key=self.api_key,
                api_version=api_version,
                azure_endpoint=self.base_url,
            )
        elif self.inference_mode == "vllm":
            self.model, self.tokenizer, self.generation_config = self.load_model(
                self.args
            )
        elif self.inference_mode == "api_async":
            self.api_key = self.args.api_key
            self.base_url = self.args.base_url
            self.model = self.args.llm_in_use
            self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        else:
            raise ValueError(f"Invalid inference mode: {self.inference_mode}")

        return self

    def unload(self):
        if self.inference_mode == "vllm":
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            destroy_model_parallel()
            destroy_distributed_environment()
            del self.model.llm_engine.model_executor.driver_worker
            del self.model
            with contextlib.suppress(AssertionError):
                torch.distributed.destroy_process_group()
            gc.collect()
            torch.cuda.empty_cache()
            import ray

            ray.shutdown()
        else:
            # For API, no explicit shutdown is needed
            if (
                self.inference_mode == "api_self_hosted"
                and getattr(self, "process", None) is not None
            ):
                stop_vllm(self.process)

    def load_model(self, args):
        # This is only for loading the model in the 'vllm' mode
        model = vllm.LLM(
            model=args.llm_in_use,
            tensor_parallel_size=torch.cuda.device_count(),
            distributed_executor_backend="ray",
            enable_prefix_caching=True,
            max_model_len=args.max_tokens,
            # max_seq_len_to_capture=8192,
            gpu_memory_utilization=0.95,
            max_num_seqs=args.max_batch_size,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.llm_in_use)
        generation_config = GenerationConfig.from_pretrained(args.llm_in_use)
        return model, tokenizer, generation_config

    def chat(self, prompt, json_model: BaseModel = None, no_tqdm=False, **kwargs):
        # Here, we assume that the prompt is a list of messages
        # and json_model is a pydantic model for validation
        self.chat_history.extend(prompt)
        for i in range(self.max_retry):
            if (
                self.inference_mode == "api"
                or self.inference_mode == "azure"
                or self.inference_mode == "api_self_hosted"
            ):
                response = asyncio.run(self.generate_async_api(
                    [self.chat_history], json_model, no_tqdm=no_tqdm, **kwargs
                ))
                if response["responses"] is None or len(response["responses"]) == 0:
                    logger.error(
                        f"Failed to get response from API: {response}, retrying..."
                    )
                    continue
                message = response["responses"][0]
                self.chat_history.append(
                    {"role": "assistant", "content": message["solution"]}
                )
                return message
            elif self.inference_mode == "api_async":
                response = asyncio.run(self.generate_async_api(
                    [self.chat_history], json_model, no_tqdm=no_tqdm, **kwargs
                ))
                if response["responses"] is None or len(response["responses"]) == 0:
                    logger.error(f"Failed to get response from API: {response}")
                    continue
                message = response["responses"][0]
                self.chat_history.append(
                    {"role": "assistant", "content": message["solution"]}
                )
                return message
            elif self.inference_mode == "vllm":
                response = self.generate_vllm(
                    [self.chat_history], json_model, no_tqdm=no_tqdm, **kwargs
                )
                if response["responses"] is None or len(response["responses"]) == 0:
                    logger.error(f"Failed to get response from API: {response}")
                    continue
                message = response["responses"][0]
                self.chat_history.append(
                    {"role": "assistant", "content": message["solution"]}
                )
                return message
            else:
                raise ValueError(f"Invalid inference mode: {self.inference_mode}")
        logger.error(f"Failed to get response from API after {self.max_retry} retries.")
        # self.chat_history.append({"role": "assistant", "content": ""})
        return {"solution": "", "reasoning": None}

    def generate(self, prompts, json_model: BaseModel = None, no_tqdm=False, **kwargs):
        if (
            self.inference_mode == "api"
            or self.inference_mode == "azure"
            or self.inference_mode == "api_self_hosted"
        ):
            return asyncio.run(self.generate_async_api(prompts, json_model, no_tqdm=no_tqdm, **kwargs))
        elif self.inference_mode == "api_async":
            return asyncio.run(
                self.generate_async_api(prompts, json_model, no_tqdm=no_tqdm, **kwargs)
            )
        elif self.inference_mode == "vllm":
            return self.generate_vllm(prompts, json_model, no_tqdm=no_tqdm, **kwargs)
        else:
            raise ValueError(f"Invalid inference mode: {self.inference_mode}")

    def generate_api(
        self, prompts, json_model: BaseModel = None, no_tqdm=False, **kwargs
    ):
        def generate_one_sample(prompt):
            if json_model is None:
                try:
                    completion = self.client.chat.completions.create(
                        model=self.model,
                        messages=prompt,
                        extra_body={"add_generation_prompt": True},
                    )
                    logger.debug(f"API response: {completion}")
                    message = completion.choices[0].message.content
                    reasoning_content = getattr(
                        completion.choices[0].message, "reasoning_content", None
                    )
                    if message is None:
                        logger.error(
                            f"Failed to get response from API: {completion}, maybe because invalid reasoning format"
                        )
                        if reasoning_content:
                            logger.error(
                                f"Reasoning content: {reasoning_content} is not None, set message as reasoning content"
                            )
                            message = reasoning_content
                            reasoning_content = None
                    return message, reasoning_content
                except Exception as e:
                    logger.error(f"Failed to get response from API: {e}")
                    return None, None
            else:
                if self.fast_mode:
                    completion = self.client.chat.completions.create(
                        model=self.model, messages=prompt, **kwargs
                    )
                    message = completion.choices[0].message.content
                    reasoning_content = getattr(
                        completion.choices[0].message, "reasoning_content", None
                    )
                    result = validate_and_parse_json_output(message, json_model)
                    if result is not None:
                        return result, reasoning_content
                    logger.info(
                        f"Failed to validate JSON for unguided decoding, turning to guided decoding. {message}"
                    )
                try:
                    completion = self.client.beta.chat.completions.parse(
                        model=self.model,
                        messages=prompt,
                        response_format=json_model,
                        extra_body=dict(guided_decoding_backend="outlines"),
                    )
                    message = completion.choices[0].message
                    logger.info(
                        f"Running guided decoding with output: {message.parsed}"
                    )
                    assert message.parsed
                    reasoning_content = getattr(
                        completion.choices[0].message, "reasoning_content", None
                    )

                    return message.content, reasoning_content
                except Exception as e:
                    logger.error(f"Failed to parse JSON: {e}, {message}")
                    return None, None

        responses = []
        success_indices = []
        failed_indices = []
        for i, prompt in tqdm.tqdm(
            enumerate(prompts), total=len(prompts), disable=no_tqdm
        ):
            output, reasoning_content = generate_one_sample(prompt)
            if output is not None:
                responses.append({"solution": output, "reasoning": reasoning_content})
                success_indices.append(i)
            else:
                failed_indices.append(i)

        return {
            "responses": responses,
            "success_indices": success_indices,
            "failed_indices": failed_indices,
        }

    async def generate_async_api(
        self, prompts, json_model: BaseModel = None, no_tqdm=False, **kwargs
    ):
        async def generate_one_sample(prompt, **kwargs):
            if json_model is None:
                try:
                    extra_body = {"add_generation_prompt": True}
                    external_extra_body_args = kwargs.get("extra_body", None)
                    if external_extra_body_args:
                        extra_body.update(external_extra_body_args)
                    # logger.debug(f"API response: {completion}")
                    completion = await self.client.chat.completions.create(
                        model=self.model,
                        messages=prompt,
                        extra_body=extra_body,
                    )
                    logger.debug(f"API response: {completion}")
                    message = completion.choices[0].message.content
                    reasoning_content = getattr(
                        completion.choices[0].message, "reasoning_content", None
                    )
                    if message is None:
                        logger.error(
                            f"Failed to get response from API: {completion}, maybe because invalid reasoning format"
                        )
                        if reasoning_content:
                            logger.error(
                                f"Reasoning content: {reasoning_content} is not None, set message as reasoning content"
                            )
                            message = reasoning_content
                            reasoning_content = None
                    return message, reasoning_content
                except Exception as e:
                    logger.error(f"Failed to get response from API: {e}")
                    return None, None
            else:
                if self.fast_mode:
                    completion = await self.client.chat.completions.create(
                        model=self.model, messages=prompt, **kwargs
                    )
                    message = completion.choices[0].message.content
                    reasoning_content = getattr(
                        completion.choices[0].message, "reasoning_content", None
                    )
                    result = validate_and_parse_json_output(message, json_model)
                    if result is not None:
                        return result, reasoning_content
                    logger.info(
                        f"Failed to validate JSON for unguided decoding, turning to guided decoding. {message}"
                    )
                try:
                    completion = await self.client.beta.chat.completions.parse(
                        model=self.model,
                        messages=prompt,
                        response_format=json_model,
                        extra_body=dict(guided_decoding_backend="outlines"),
                    )
                    message = completion.choices[0].message
                    logger.info(
                        f"Running guided decoding with output: {message.parsed}"
                    )
                    assert message.parsed
                    reasoning_content = getattr(
                        completion.choices[0].message, "reasoning_content", None
                    )

                    return message.content, reasoning_content
                except Exception as e:
                    logger.error(f"Failed to parse JSON: {e}, {message}")
                    return None, None

        sem = Semaphore(self.max_batch_size)

        async def generate_one_sample_limited(prompt, **kwargs):
            async with sem:
                return await generate_one_sample(prompt, **kwargs)

        tasks = [generate_one_sample_limited(prompt, **kwargs) for prompt in prompts]
        # Use asyncio.gather to run the tasks concurrently
        # and wait for all of them to finish
        responses = await tqdm_asyncio.gather(*tasks)
        # Process the results
        success_indices = []
        failed_indices = []
        results = []
        for i, (output, reasoning_content) in enumerate(responses):
            if output is not None:
                results.append({"solution": output, "reasoning": reasoning_content})
                success_indices.append(i)
            else:
                failed_indices.append(i)
        return {
            "responses": results,
            "success_indices": success_indices,
            "failed_indices": failed_indices,
        }

    def generate_async_api_batched(
        self, prompts, json_model: BaseModel = None, no_tqdm=False, **kwargs
    ):

        max_batch_size = kwargs.get("max_batch_size", self.max_tokens)

        final_results = {
            "responses": [],
            "success_indices": [],
            "failed_indices": [],
        }
        batched_samples = []
        for i, prompt in tqdm.tqdm(
            enumerate(prompts), total=len(prompts), disable=no_tqdm
        ):
            batched_samples.append(prompt)  # Pass the prompt to the function
            if len(batched_samples) >= max_batch_size:
                results = asyncio.run(
                    self.generate_async_api(
                        batched_samples, json_model, no_tqdm=no_tqdm, **kwargs
                    )
                )
                final_results["responses"].extend(results["responses"])
                final_results["success_indices"].extend(results["success_indices"])
                final_results["failed_indices"].extend(results["failed_indices"])
                batched_samples = []  # Reset the batch
        # Process any remaining tasks
        if len(batched_samples) > 0:
            results = asyncio.run(
                self.generate_async_api(
                    batched_samples, json_model, no_tqdm=no_tqdm, **kwargs
                )
            )
            final_results["responses"].extend(results["responses"])
            final_results["success_indices"].extend(results["success_indices"])
            final_results["failed_indices"].extend(results["failed_indices"])
            batched_samples = []  # Reset the batch
        return final_results

    def generate_vllm(
        self, prompts, json_model: BaseModel = None, no_tqdm=False, **kwargs
    ):

        def setup_sampling_params(guided_decoding=None):
            # This function sets up the sampling parameters for the model
            # It prioritizes the parameters set given by the user, and falls back to default values (either from generation_config or hardcoded)
            temperature = kwargs.get(
                "temperature",
                (
                    0.7
                    if self.generation_config.temperature is None
                    else self.generation_config.temperature
                ),
            )
            top_p = kwargs.get(
                "top_p",
                (
                    0.9
                    if self.generation_config.top_p is None
                    else self.generation_config.top_p
                ),
            )
            top_k = kwargs.get(
                "top_k",
                (
                    50
                    if self.generation_config.top_k is None
                    else self.generation_config.top_k
                ),
            )
            repetition_penalty = kwargs.get(
                "repetition_penalty",
                (
                    1.05
                    if getattr(self.generation_config, "repetition_penalty", None)
                    is None
                    else self.generation_config.repetition_penalty
                ),
            )
            max_tokens = kwargs.get("max_tokens", 16384)
            guided_decoding = kwargs.get("guided_decoding", guided_decoding)
            return SamplingParams(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
                guided_decoding=guided_decoding,
                repetition_penalty=repetition_penalty,
            )

        def run_unguided_inference(prompts):
            sampling_params = setup_sampling_params()
            model_inputs = [
                self.tokenizer.apply_chat_template(
                    prompt, tokenize=False, add_generation_prompt=True
                )
                for prompt in prompts
            ]
            logger.info(f"Running unguided decoding with {len(model_inputs)} prompts")
            outputs = self.model.generate(model_inputs, sampling_params=sampling_params)
            outputs = [
                post_process_output(output.outputs[0].text) for output in outputs
            ]
            return outputs

        def run_guided_inference(prompts):
            json_schema = json_model.model_json_schema()
            guided_decoding_params = GuidedDecodingParams(json=json_schema)
            sampling_params = setup_sampling_params(guided_decoding_params)
            model_inputs = [
                self.tokenizer.apply_chat_template(
                    prompt, tokenize=False, add_generation_prompt=True
                )
                for prompt in prompts
            ]
            logger.info(f"Running guided decoding with {len(model_inputs)} prompts")
            outputs = self.model.generate(model_inputs, sampling_params=sampling_params)
            outputs = [
                post_process_output(output.outputs[0].text) for output in outputs
            ]
            return outputs

        # For efficiency purpose, by default, we first run with unguided decoding
        # and then run with guided decoding if any samples are not valid JSON

        if json_model is None:
            outputs = run_unguided_inference(prompts)
            assert len(outputs) == len(prompts)
            return {
                "responses": outputs,
                "success_indices": list(range(len(prompts))),
                "failed_indices": [],
            }

        failed_inputs = [
            (i, prompt) for i, prompt in enumerate(prompts) if prompt is None
        ]
        success_results = []
        if self.fast_mode:
            outputs = run_unguided_inference(prompts)
            assert len(outputs) == len(prompts)

            # Validate JSON outputs
            success_results = []
            failed_inputs = []
            for i, output in enumerate(outputs):
                result = validate_and_parse_json_output(output, json_model)
                if result is not None:
                    success_results.append((i, result))
                else:
                    failed_inputs.append((i, prompts[i]))
            if len(failed_inputs) > 0:
                logger.info(
                    f"Failed to validate JSON for {len(failed_inputs)} samples. Will run guided decoding later."
                )

        guided_outputs = run_guided_inference([prompt for _, prompt in failed_inputs])
        assert len(guided_outputs) == len(failed_inputs)
        for (i, _), output in zip(failed_inputs, guided_outputs):
            result = validate_and_parse_json_output(output, json_model)
            if result is not None:
                success_results.append((i, result))
            else:
                logger.error(
                    f"Failed to validate JSON for guided decoding: {output} {result}"
                )

        success_results.sort(key=lambda x: x[0])
        responses = [result for _, result in success_results]
        success_indices = [i for i, _ in success_results]
        failed_indices = [i for i in range(len(prompts)) if i not in success_indices]
        return {
            "responses": responses,
            "success_indices": success_indices,
            "failed_indices": failed_indices,
        }
