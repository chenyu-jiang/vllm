from typing import List, Optional, Tuple, Any

import pytest
import torch
import argparse
from transformers import AutoModelForCausalLM

from vllm import LLM, SamplingParams
from vllm.utils import Counter, MMInputType
from vllm.engine.async_llm_engine import AsyncMLLMEngine, AsyncEngineArgs, \
                                         AsyncStream
from vllm.transformers_utils.tokenizer import get_tokenizer

_TEST_PROMPTS = [
    "vLLM is a high-throughput and memory-efficient inference and serving engine for LLMs.",
    "Briefly describe the major milestones in the development of artificial intelligence from 1950 to 2020.",
    "Compare and contrast artificial intelligence with human intelligence in terms of processing information.",
    "Describe the basic components of a neural network and how it can be trained.",
    "Write a short story about a robot that dreams for the first time.",
    "Analyze the impact of the COVID-19 pandemic on global economic structures and future business models.",
    "Explain the cultural significance of the Mona Lisa painting, and how its perception might vary in Western versus Eastern societies.",
    "Translate the following English sentence into Japanese, French, and Swahili: 'The early bird catches the worm.'",
]

_MM_TEST_PROMPTS = [
    "What are the recommended things to do at this place?",
    "What should I be careful about when I visit this place?",
    "Describe the major attractions of this place.",
    "Describe the picture.",
    "What is the name of this place?",
    "Can you write a diary about a visit to this place for me?",
    "What is the weather like as shown in the picture?",
]

def _mm_test_template(prompt: str) -> str:
    response= ("A chat between a curious human and an artificial intelligence "
               "assistant. The assistant gives helpful, detailed, and polite "
               "answers to the human's questions. USER: <image>\n"
               f"{prompt} ASSISTANT:")
    return response


@pytest.fixture
def example_prompts() -> List[str]:
    return _TEST_PROMPTS

@pytest.fixture
def mm_example_prompts() -> List[str]:
    return [_mm_test_template(prompt) for prompt in _MM_TEST_PROMPTS]


_STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.half,
    "bfloat16": torch.bfloat16,
    "float": torch.float,
}


class HfRunner:

    def __init__(
        self,
        model_name: str,
        tokenizer_name: Optional[str] = None,
        dtype: str = "half",
    ) -> None:
        assert dtype in _STR_DTYPE_TO_TORCH_DTYPE
        torch_dtype = _STR_DTYPE_TO_TORCH_DTYPE[dtype]
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        ).cuda()
        if tokenizer_name is None:
            tokenizer_name = model_name
        self.tokenizer = get_tokenizer(
            tokenizer_name,
            trust_remote_code=True,
        )

    def generate(
        self,
        prompts: List[str],
        **kwargs,
    ) -> List[Tuple[List[int], str]]:
        outputs: List[Tuple[List[int], str]] = []
        for prompt in prompts:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
            output_ids = self.model.generate(
                input_ids.cuda(),
                use_cache=True,
                **kwargs,
            )
            output_str = self.tokenizer.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            output_ids = output_ids.cpu().tolist()
            outputs.append((output_ids, output_str))
        return outputs

    def generate_greedy(
        self,
        prompts: List[str],
        max_tokens: int,
    ) -> List[Tuple[List[int], str]]:
        outputs = self.generate(prompts,
                                do_sample=False,
                                max_new_tokens=max_tokens)
        for i in range(len(outputs)):
            output_ids, output_str = outputs[i]
            outputs[i] = (output_ids[0], output_str[0])
        return outputs

    def generate_beam_search(
        self,
        prompts: List[str],
        beam_width: int,
        max_tokens: int,
    ) -> List[Tuple[List[int], str]]:
        outputs = self.generate(prompts,
                                do_sample=False,
                                max_new_tokens=max_tokens,
                                num_beams=beam_width,
                                num_return_sequences=beam_width)
        for i in range(len(outputs)):
            output_ids, output_str = outputs[i]
            for j in range(len(output_ids)):
                output_ids[j] = [
                    x for x in output_ids[j]
                    if x != self.tokenizer.pad_token_id
                ]
            outputs[i] = (output_ids, output_str)
        return outputs

    def generate_greedy_logprobs(
        self,
        prompts: List[str],
        max_tokens: int,
    ) -> List[List[torch.Tensor]]:
        all_logprobs = []
        for prompt in prompts:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
            output = self.model.generate(
                input_ids.cuda(),
                use_cache=True,
                do_sample=False,
                max_new_tokens=max_tokens,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            seq_logprobs = []
            for hidden_states in output.hidden_states:
                last_hidden_states = hidden_states[-1][0]
                logits = torch.matmul(
                    last_hidden_states,
                    self.model.get_output_embeddings().weight.t(),
                )
                if self.model.get_output_embeddings().bias is not None:
                    logits += self.model.get_output_embeddings(
                    ).bias.unsqueeze(0)
                logprobs = torch.nn.functional.log_softmax(logits,
                                                           dim=-1,
                                                           dtype=torch.float32)
                seq_logprobs.append(logprobs)
            all_logprobs.append(seq_logprobs)
        return all_logprobs


@pytest.fixture
def hf_runner():
    return HfRunner


class VllmRunner:

    def __init__(
        self,
        model_name: str,
        tokenizer_name: Optional[str] = None,
        dtype: str = "half",
    ) -> None:
        self.model = LLM(
            model=model_name,
            tokenizer=tokenizer_name,
            trust_remote_code=True,
            dtype=dtype,
            swap_space=0,
        )

    def generate(
        self,
        prompts: List[str],
        sampling_params: SamplingParams,
        prompt_embeds: List[torch.Tensor] = None,
    ) -> List[Tuple[List[int], str]]:
        req_outputs = self.model.generate(prompts,
                                          sampling_params=sampling_params,
                                          prompt_embeds=prompt_embeds)
        outputs = []
        for req_output in req_outputs:
            prompt_str = req_output.prompt
            prompt_ids = req_output.prompt_token_ids
            req_sample_output_ids = []
            req_sample_output_strs = []
            for sample in req_output.outputs:
                output_str = sample.text
                output_ids = sample.token_ids
                req_sample_output_ids.append(prompt_ids + output_ids)
                req_sample_output_strs.append(prompt_str + output_str)
            outputs.append((req_sample_output_ids, req_sample_output_strs))
        return outputs

    def generate_greedy(
        self,
        prompts: List[str],
        max_tokens: int,
        prompt_embeds: List[torch.Tensor] = None,
    ) -> List[Tuple[List[int], str]]:
        greedy_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        outputs = self.generate(prompts,
                                greedy_params,
                                prompt_embeds=prompt_embeds)
        return [(output_ids[0], output_str[0])
                for output_ids, output_str in outputs]

    def generate_beam_search(
        self,
        prompts: List[str],
        beam_width: int,
        max_tokens: int,
        prompt_embeds: List[torch.Tensor] = None,
    ) -> List[Tuple[List[int], str]]:
        beam_search_params = SamplingParams(n=beam_width,
                                            use_beam_search=True,
                                            temperature=0.0,
                                            max_tokens=max_tokens)
        outputs = self.generate(prompts,
                                beam_search_params,
                                prompt_embeds=prompt_embeds)
        return outputs

class MultimodalVllmRunner:
    def __init__(
        self,
        model_name: str,
        tokenizer_name: Optional[str] = None,
        dtype: str = "half",
    ) -> None:
        parser = argparse.ArgumentParser()
        parser = AsyncEngineArgs.add_cli_args(parser)
        args = parser.parse_args([])

        args.model = model_name
        args.tokenizer = tokenizer_name
        args.swap_space = 0
        args.trust_remote_code = True
        args.dtype = dtype

        engine_args = AsyncEngineArgs.from_cli_args(args)
        self.async_engine = AsyncMLLMEngine.from_engine_args(engine_args)
        self.request_counter = Counter()

    async def generate(
        self,
        prompts: List[str],
        sampling_params: SamplingParams,
        modality_inputs: List[List[Tuple[MMInputType, Any]]] = None,
    ) -> List[Tuple[List[int], str]]:
        if modality_inputs is not None:
            assert len(prompts) == len(modality_inputs)
        req_outputs: List[AsyncStream] = []
        for prompt, mm_inputs in zip(prompts, modality_inputs):
            request_id = str(next(self.request_counter))
            req_output = await self.async_engine.add_request(request_id,
                                                        prompt,
                                                        sampling_params,
                                                        modality_inputs=mm_inputs)
            req_outputs.append(req_output)
        outputs = []
        for req_stream in req_outputs:
            final_output = None
            async for request_output in req_stream:
                final_output = request_output
            prompt_str = final_output.prompt
            prompt_ids = final_output.prompt_token_ids
            req_sample_output_ids = []
            req_sample_output_strs = []
            for sample in final_output.outputs:
                output_str = sample.text
                output_ids = sample.token_ids
                req_sample_output_ids.append(prompt_ids + output_ids)
                req_sample_output_strs.append(prompt_str + output_str)
                req_sample_output_strs.append(output_str)
            outputs.append((req_sample_output_ids, req_sample_output_strs))
        return outputs

    async def generate_greedy(
        self,
        prompts: List[str],
        max_tokens: int,
        modality_inputs: List[List[Tuple[MMInputType, Any]]] = None,
    ) -> List[Tuple[List[int], str]]:
        greedy_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        outputs = await self.generate(prompts,
                                greedy_params,
                                modality_inputs=modality_inputs)
        return [(output_ids[0], output_str[0])
                for output_ids, output_str in outputs]

    async def generate_beam_search(
        self,
        prompts: List[str],
        beam_width: int,
        max_tokens: int,
        modality_inputs: List[List[Tuple[MMInputType, Any]]] = None,
    ) -> List[Tuple[List[int], str]]:
        beam_search_params = SamplingParams(n=beam_width,
                                            use_beam_search=True,
                                            temperature=0.0,
                                            max_tokens=max_tokens)
        outputs = await self.generate(prompts,
                                beam_search_params,
                                modality_inputs=modality_inputs)
        return outputs


@pytest.fixture
def vllm_runner():
    return VllmRunner


@pytest.fixture
def mm_vllm_runner():
    return MultimodalVllmRunner
