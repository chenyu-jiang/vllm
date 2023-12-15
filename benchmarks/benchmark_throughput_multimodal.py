"""Benchmark offline inference throughput."""
import os
import argparse
import json
import random
import time
from typing import List, Optional, Tuple

import torch
from transformers import AutoTokenizer
from tqdm import tqdm

def _mm_test_template(prompt: str) -> str:
    response= ("A chat between a curious human and an artificial intelligence "
               "assistant. The assistant gives helpful, detailed, and polite "
               "answers to the human's questions. USER: <image>\n"
               f"{prompt} ASSISTANT:")
    return response

def run_vllm(
    requests: List[Tuple[str, str, int]],
    model: str,
    tokenizer: str,
    quantization: Optional[str],
    tensor_parallel_size: int,
    seed: int,
    n: int,
    use_beam_search: bool,
    trust_remote_code: bool,
    dtype: str,
    max_model_len: Optional[int] = None,
    encoder_max_batch_size: int = 4,
    encoder_max_pending_requests: int = 16,
) -> float:
    from vllm import AsyncMLLMEngine, SamplingParams, AsyncEngineArgs
    from vllm.utils import MMInputType
    from vllm.engine.async_llm_engine import AsyncStream
    import argparse
    import asyncio
    from asyncio import Task

    parser = argparse.ArgumentParser()
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args([])

    args.model = model
    args.tokenizer = tokenizer
    args.quantization = quantization
    args.tensor_parallel_size = tensor_parallel_size
    args.seed = seed
    args.swap_space = 4
    args.trust_remote_code = trust_remote_code
    args.dtype = dtype
    args.max_model_len = max_model_len
    args.disable_log_stats = False
    args.disable_log_requests = True

    engine_args = AsyncEngineArgs.from_cli_args(args)
    async_engine = AsyncMLLMEngine.from_engine_args(engine_args,
                                                    encoder_max_batch_size=encoder_max_batch_size,
                                                    encoder_max_pending_requests=encoder_max_pending_requests)

    total_generated_tokens = 0
    # Add the requests to the engine.
    async def _run():
        nonlocal total_generated_tokens
        tasks: List[Task[AsyncStream]] = []
        for req_id, (prompt, image, output_len) in enumerate(requests):
            sampling_params = SamplingParams(
                n=n,
                temperature=0.0 if use_beam_search else 1.0,
                top_p=1.0,
                use_beam_search=use_beam_search,
                ignore_eos=True,
                max_tokens=output_len,
            )
            modality_inputs = [(MMInputType.IMAGE, image)]
            task = asyncio.create_task(async_engine.add_request(
                                        req_id,
                                        prompt,
                                        sampling_params,
                                        modality_inputs=modality_inputs))
            tasks.append(task)
        await asyncio.gather(*tasks, return_exceptions=True)
        outputs = []
        for t in tasks:
            req_stream = t.result()
            final_output = None
            async for request_output in req_stream:
                final_output = request_output
        #     prompt_str = final_output.prompt
        #     prompt_ids = final_output.prompt_token_ids
        #     req_sample_output_ids = []
        #     req_sample_output_strs = []
        #     for sample in final_output.outputs:
        #         output_str = sample.text
        #         output_ids = sample.token_ids
        #         req_sample_output_ids.append(prompt_ids + output_ids)
        #         req_sample_output_strs.append(prompt_str + output_str)
        #         total_generated_tokens += len(output_ids)
        #     outputs.append((req_sample_output_ids, req_sample_output_strs))
        # print("Output:", outputs[0][1][0])
        return outputs

    start = time.perf_counter()
    asyncio.run(_run())
    end = time.perf_counter()
    return end - start, total_generated_tokens


def main(args: argparse.Namespace):
    torch.cuda.memory._record_memory_history(max_entries=100000)
    print(args)
    random.seed(args.seed)

    # Sample the requests.
    if args.dataset is None:
        # Synthesize a prompt with the given input length.
        prompt = _mm_test_template(args.prompt)
        image = args.image
        if not os.path.exists(image):
            raise ValueError(f"Image {image} does not exist on disk.")
        requests = [(prompt, image, args.output_len)
                    for _ in range(args.num_prompts)]
    else:
        raise NotImplementedError("Using dataset is not supported yet")
        # requests = sample_requests(args.dataset, args.num_prompts, tokenizer,
        #                            args.output_len)

    elapsed_time, total_generated_tokens = run_vllm(requests, args.model, args.tokenizer,
                            args.quantization, args.tensor_parallel_size,
                            args.seed, args.n, args.use_beam_search,
                            args.trust_remote_code, args.dtype,
                            args.max_model_len)
    total_num_tokens = sum(output_len
                           for _, _, output_len in requests)
    print("Elapsed time: {:.2f} s".format(elapsed_time))
    print(f"Throughput: {len(requests) / elapsed_time:.2f} requests/s, "
          f"{total_num_tokens / elapsed_time:.2f} tokens/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the throughput.")
    parser.add_argument("--dataset",
                        type=str,
                        default=None,
                        help="Path to the dataset.")
    # parser.add_argument("--input-len",
    #                     type=int,
    #                     default=None,
    #                     help="Input prompt length for each request")
    parser.add_argument("--image",
                        type=str,
                        default=None,
                        help="Path to the image file.")
    parser.add_argument("--prompt",
                        type=str,
                        default=None,
                        help="Prompt to use for benchmark.")
    parser.add_argument("--output-len",
                        type=int,
                        default=None,
                        help="Output length for each request. Overrides the "
                        "output length from the dataset.")
    parser.add_argument("--model", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument('--quantization',
                        '-q',
                        choices=['awq', 'squeezellm', None],
                        default=None)
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
    parser.add_argument("--n",
                        type=int,
                        default=1,
                        help="Number of generated sequences per prompt.")
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--num-prompts",
                        type=int,
                        default=1000,
                        help="Number of prompts to process.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--trust-remote-code',
                        action='store_true',
                        help='trust remote code from huggingface')
    parser.add_argument(
        '--max-model-len',
        type=int,
        default=None,
        help='Maximum length of a sequence (including prompt and output). '
        'If None, will be derived from the model.')
    parser.add_argument(
        '--dtype',
        type=str,
        default='auto',
        choices=['auto', 'half', 'float16', 'bfloat16', 'float', 'float32'],
        help='data type for model weights and activations. '
        'The "auto" option will use FP16 precision '
        'for FP32 and FP16 models, and BF16 precision '
        'for BF16 models.')
    args = parser.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.model
    if args.dataset is None:
        assert args.image is not None
        assert args.prompt is not None
        # assert args.input_len is not None
        assert args.output_len is not None
    else:
        raise NotImplementedError("Using dataset is not supported yet")
        # assert args.input_len is None
    # TODO: add back the disabled features
    assert args.quantization is None, "Quantization is not supported yet"
    main(args)
