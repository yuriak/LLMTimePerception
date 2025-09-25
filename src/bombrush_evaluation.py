from grid_world import GridWorldEnv
from llm import LLM
import os
from multiprocessing import Pool
import subprocess
import time
import requests
import signal
import sys
import os
import logging

logger = logging.getLogger(__name__)

def run_exp(env_info):
    """
    Run an experiment with the given environment.
    """
    env, output_dir = env_info
    env.llm.initialize()
    env.reset()
    env.run_simulation()
    output_dir = os.path.join(output_dir, f"eval_{env.seed}.pkl")
    env.save_results(output_dir)
    logger.info(f"Evaluation completed for seed {env.seed}. Results saved to {output_dir}.")
    del env

def start_vllm(llm_in_use, max_tokens=16384):
    env = os.environ.copy()
    env["VLLM_USE_V1"] = "0"
    reasoning_model = True if any([ u in llm_in_use for u in ["deepseek", "QWQ"]]) else False
    model_arg = f"--model {llm_in_use}"
    if reasoning_model:
        model_arg += " --reasoning-parser deepseek_r1 --enable-reasoning"

    vllm_cmd=f"""
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
        preexec_fn=os.setsid  # 为了后续使用pgid杀掉整个进程组
    )
    logger.info(f"vLLM server started with PID {process.pid}.")
    return process

def wait_for_vllm_ready(port):
    logging.info("waiting for vLLM server to be ready...")
    url = f"http://localhost:{port}/v1/models"
    for _ in range(60*15): # 15 minutes timeout
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

class Evaluation:

    @classmethod
    def add_arguments(cls, parser):
        """
        Add command line arguments for the Evaluation class.
        """
        # Add arguments for the LLM
        LLM.add_arguments(parser)

        # Add arguments for the DynamicEnv
        GridWorldEnv.add_arguments(parser)

        parser.add_argument(
            "--max_runs",
            type=int,
            default=50,
            help="Maximum number of runs for evaluation."
        )
        parser.add_argument(
            "--num_workers",
            type=int,
            default=2,
            help="Maximum number of threads for evaluation."
        )
        parser.add_argument(
            "--output_dir",
            type=str,
            default="eval_results",
            help="Type of environment to use for evaluation."
        )
        parser.add_argument(
            "--self_serve_vllm",
            action="store_true",
            help="Whether to use self-serve vLLM."
        )

    def __init__(self, args):
        """
        Initialize the Evaluation class with the given arguments.
        """
        self.max_runs = args.max_runs
        self.num_workers = args.num_workers
        self.output_dir = args.output_dir
        debug = getattr(args, "debug", False)
        if debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
        os.makedirs(self.output_dir, exist_ok=True)
        # Initialize all the environments and LLMs
        self.args = args


    def evaluate(self):
        if self.num_workers > 1:
            envs = []
            for i in range(self.max_runs):
                assert getattr(self.args, "inference_mode", None) in ["api","azure"], "Inference mode should be api or azure"
                llm = LLM(self.args)
                seed = i
                env = GridWorldEnv(self.args, llm, seed=seed)
                envs.append((env, self.output_dir))
            with Pool(self.num_workers) as pool:
                pool.map(run_exp, envs)
        else:
            self.evaluate_single_process()
    
    def evaluate_single_process(self):
        for i in range(self.max_runs):
            llm = LLM(self.args)
            env = GridWorldEnv(self.args, llm, seed=i)
            env.llm.initialize()
            env.reset()
            env.run_simulation()
            output_dir = os.path.join(output_dir, f"eval_{env.seed}.pkl")
            env.save_results(output_dir)
            print(f"Evaluation completed for seed {env.seed}. Results saved to {output_dir}.")
            del env


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    Evaluation.add_arguments(parser)
    args = parser.parse_args()
    evaluator = Evaluation(args)
    if args.self_serve_vllm:
        print("Starting vLLM server...")
        process = start_vllm(args.llm_in_use, args.max_tokens)
        if wait_for_vllm_ready(9876):
            evaluator.evaluate()
        stop_vllm(process)
    else:
        print("Using existing vLLM server...")
        evaluator.evaluate()
    print("Evaluation completed.")
    