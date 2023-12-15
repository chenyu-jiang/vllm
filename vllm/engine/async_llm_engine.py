import asyncio
import traceback
import time
from functools import partial
from typing import (Any, Dict, Iterable, List, Optional, Set, Tuple, Type,
                    Union, Callable)
import torch

from vllm.utils import MMInputType
from vllm.config import ModelConfig
from vllm.model_executor import get_multimodal_encoder
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.engine.ray_utils import initialize_cluster, ray
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams

logger = init_logger(__name__)


class AsyncEngineDeadError(RuntimeError):
    pass


def _raise_exception_on_finish(task: asyncio.Task,
                               request_tracker: "RequestTracker") -> None:
    msg = ("Task finished unexpectedly. This should never happen! "
           "Please open an issue on Github.")
    try:
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except Exception as exc:
            raise AsyncEngineDeadError(
                msg + " See stack trace above for the actual cause.") from exc
        raise AsyncEngineDeadError(msg)
    except Exception as exc:
        request_tracker.propagate_exception(exc)
        raise exc

def _raise_exception_on_finish_encoders(task: asyncio.Task) -> None:
    msg = ("Task finished unexpectedly. This should never happen! "
           "Please open an issue on Github.")
    try:
        task.result()
    except asyncio.CancelledError:
        return
    except Exception as exc:
        traceback.print_exception(None, exc, exc.__traceback__)
        exit(1)
    raise AsyncEngineDeadError(msg)


class AsyncStream:
    """A stream of RequestOutputs for a request that can be
    iterated over asynchronously."""

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self._queue = asyncio.Queue()
        self._finished = False

    def put(self, item: RequestOutput) -> None:
        if self._finished:
            return
        self._queue.put_nowait(item)

    def finish(self) -> None:
        self._queue.put_nowait(StopIteration)
        self._finished = True

    @property
    def finished(self) -> bool:
        return self._finished

    def __aiter__(self):
        return self

    async def __anext__(self) -> RequestOutput:
        result = await self._queue.get()
        if result is StopIteration:
            raise StopAsyncIteration
        elif isinstance(result, Exception):
            raise result
        return result


class RequestTracker:
    """Synchronous abstraction for tracking requests."""

    def __init__(self) -> None:
        self._request_streams: Dict[str, AsyncStream] = {}
        self._finished_requests: asyncio.Queue[str] = asyncio.Queue()
        self._new_requests: asyncio.Queue[Tuple[AsyncStream,
                                                dict]] = asyncio.Queue()
        self.new_requests_event = None

    def __contains__(self, item):
        return item in self._request_streams

    def init_event(self):
        self.new_requests_event = asyncio.Event()

    def propagate_exception(self,
                            exc: Exception,
                            request_id: Optional[str] = None) -> None:
        """Propagate an exception to request streams
        (all if request_id is None)."""
        if request_id is not None:
            self._request_streams[request_id].put(exc)
        else:
            for stream in self._request_streams.values():
                stream.put(exc)

    def process_request_output(self,
                               request_output: RequestOutput,
                               *,
                               verbose: bool = False) -> None:
        """Process a request output from the engine."""
        request_id = request_output.request_id

        self._request_streams[request_id].put(request_output)
        if request_output.finished:
            if verbose:
                logger.info(f"Finished request {request_id}.")
            self.abort_request(request_id)

    def add_request(self, request_id: str,
                    **engine_add_request_kwargs) -> AsyncStream:
        """Add a request to be sent to the engine on the next background
        loop iteration."""
        if request_id in self._request_streams:
            raise KeyError(f"Request {request_id} already exists.")

        stream = AsyncStream(request_id)
        self._new_requests.put_nowait((stream, {
            "request_id": request_id,
            **engine_add_request_kwargs
        }))

        self.new_requests_event.set()

        return stream

    def abort_request(self, request_id: str, *, verbose: bool = False) -> None:
        """Abort a request during next background loop iteration."""
        if verbose:
            logger.info(f"Aborted request {request_id}.")

        self._finished_requests.put_nowait(request_id)

        if request_id not in self._request_streams or self._request_streams[
                request_id].finished:
            # The request has already finished or been aborted.
            return

        self._request_streams[request_id].finish()

    def get_new_and_finished_requests(self) -> Tuple[List[Dict], Set[str]]:
        """Get the new requests and finished requests to be
        sent to the engine."""
        new_requests: List[Dict] = []
        finished_requests: Set[str] = set()

        while not self._finished_requests.empty():
            request_id = self._finished_requests.get_nowait()
            finished_requests.add(request_id)
            self._request_streams.pop(request_id, None)

        while not self._new_requests.empty():
            stream, new_request = self._new_requests.get_nowait()
            if stream.request_id in finished_requests:
                # The request has already been aborted.
                stream.finish()
                continue
            self._request_streams[stream.request_id] = stream
            new_requests.append(new_request)

        self.new_requests_event.clear()

        return new_requests, finished_requests

    async def wait_for_new_requests(self):
        await self.new_requests_event.wait()


class _AsyncLLMEngine(LLMEngine):
    """Extension of LLMEngine to add async methods."""

    async def step_async(self) -> List[RequestOutput]:
        """Performs one decoding iteration and returns newly generated results.
        The workers are ran asynchronously if possible.

        This function performs one decoding iteration of the engine. It first
        schedules the sequences to be executed in the next iteration and the
        token blocks to be swapped in/out/copy. Then, it executes the model
        and updates the scheduler with the model outputs. Finally, it decodes
        the sequences and returns the newly generated results.
        """
        seq_group_metadata_list, scheduler_outputs, ignored = self._schedule()
        if scheduler_outputs.is_empty():
            return ignored

        # Execute the model.
        output = await self._run_workers_async(
            "execute_model",
            seq_group_metadata_list=seq_group_metadata_list,
            blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
            blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
            blocks_to_copy=scheduler_outputs.blocks_to_copy,
        )

        return self._process_model_outputs(output, scheduler_outputs) + ignored

    async def embed_inputs_async(self, input_ids: List[int]) -> torch.Tensor:
        """Embeds the inputs."""
        output = await self._run_workers_async("embed_inputs", input_ids=input_ids)
        return output

    async def _run_workers_async(
        self,
        method: str,
        *args,
        get_all_outputs: bool = False,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers."""
        coros = []
        for worker in self.workers:
            if self.parallel_config.worker_use_ray:
                coros.append(
                    worker.execute_method.remote(method, *args, **kwargs))
            else:
                executor = getattr(worker, method)
                coros.append(asyncio.get_event_loop().run_in_executor(
                    None, partial(executor, *args, **kwargs)))

        all_outputs = await asyncio.gather(*coros)

        if get_all_outputs:
            return all_outputs

        # Make sure all workers have the same results.
        output = all_outputs[0]
        for other_output in all_outputs[1:]:
            assert output == other_output
        return output


class AsyncLLMEngine:
    """An asynchronous wrapper for LLMEngine.

    This class is used to wrap the LLMEngine class to make it asynchronous. It
    uses asyncio to create a background loop that keeps processing incoming
    requests. The LLMEngine is kicked by the generate method when there
    are requests in the waiting queue. The generate method yields the outputs
    from the LLMEngine to the caller.

    NOTE: For the comprehensive list of arguments, see `LLMEngine`.

    Args:
        worker_use_ray: Whether to use Ray for model workers. Required for
            distributed execution. Should be the same as
            `parallel_config.worker_use_ray`.
        engine_use_ray: Whether to make LLMEngine a Ray actor. If so, the
            async frontend will be executed in a separate process as the
            model workers.
        log_requests: Whether to log the requests.
        start_engine_loop: If True, the background task to run the engine
            will be automatically started in the generate call.
        *args, *kwargs: Arguments for LLMEngine.
    """

    _engine_class: Type[_AsyncLLMEngine] = _AsyncLLMEngine

    def __init__(self,
                 worker_use_ray: bool,
                 engine_use_ray: bool,
                 *args,
                 log_requests: bool = True,
                 max_log_len: Optional[int] = None,
                 start_engine_loop: bool = True,
                 **kwargs) -> None:
        self.worker_use_ray = worker_use_ray
        self.engine_use_ray = engine_use_ray
        self.log_requests = log_requests
        self.max_log_len = max_log_len
        self.engine = self._init_engine(*args, **kwargs)

        self.background_loop = None
        # We need to keep a reference to unshielded
        # task as well to prevent it from being garbage
        # collected
        self._background_loop_unshielded = None
        self.start_engine_loop = start_engine_loop
        self._request_tracker = RequestTracker()

    @property
    def is_running(self) -> bool:
        return (self.background_loop is not None
                and not self.background_loop.done())

    def start_background_loop(self) -> None:
        """Start the background loop."""
        if self.is_running:
            raise RuntimeError("Background loop is already running.")
        self._request_tracker.init_event()

        self._background_loop_unshielded = asyncio.get_event_loop(
        ).create_task(self.run_engine_loop())
        self._background_loop_unshielded.add_done_callback(
            partial(_raise_exception_on_finish,
                    request_tracker=self._request_tracker))
        self.background_loop = asyncio.shield(self._background_loop_unshielded)

    def _init_engine(self, *args,
                     **kwargs) -> Union[_AsyncLLMEngine, "ray.ObjectRef"]:
        if not self.engine_use_ray:
            engine_class = self._engine_class
        elif self.worker_use_ray:
            engine_class = ray.remote(num_cpus=0)(self._engine_class).remote
        else:
            # FIXME(woosuk): This is a bit hacky. Be careful when changing the
            # order of the arguments.
            cache_config = args[1]
            parallel_config = args[2]
            if parallel_config.tensor_parallel_size == 1:
                num_gpus = cache_config.gpu_memory_utilization
            else:
                num_gpus = 1
            engine_class = ray.remote(num_gpus=num_gpus)(
                self._engine_class).remote
        return engine_class(*args, **kwargs)

    async def engine_step(self) -> bool:
        """Kick the engine to process the waiting requests.

        Returns True if there are in-progress requests."""

        new_requests, finished_requests = (
            self._request_tracker.get_new_and_finished_requests())

        for new_request in new_requests:
            # Add the request into the vLLM engine's waiting queue.
            # TODO: Maybe add add_request_batch to reduce Ray overhead
            if self.engine_use_ray:
                await self.engine.add_request.remote(**new_request)
            else:
                self.engine.add_request(**new_request)

        if finished_requests:
            await self._engine_abort(finished_requests)

        if self.engine_use_ray:
            request_outputs = await self.engine.step.remote()
        else:
            request_outputs = await self.engine.step_async()

        # Put the outputs into the corresponding streams.
        for request_output in request_outputs:
            self._request_tracker.process_request_output(
                request_output, verbose=self.log_requests)

        return len(request_outputs) > 0

    async def _engine_abort(self, request_ids: Iterable[str]):
        if self.engine_use_ray:
            await self.engine.abort_request.remote(request_ids)
        else:
            self.engine.abort_request(request_ids)

    async def run_engine_loop(self):
        # Initialize the RequestTracker here so it uses the right event loop.
        has_requests_in_progress = False
        while True:
            if not has_requests_in_progress:
                await self._request_tracker.wait_for_new_requests()
            has_requests_in_progress = await self.engine_step()
            await asyncio.sleep(0)

    async def add_request(
        self,
        request_id: str,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[List[int]] = None,
        arrival_time: Optional[float] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
    ) -> AsyncStream:
        if self.log_requests:
            shortened_prompt = prompt
            shortened_token_ids = prompt_token_ids
            if self.max_log_len is not None:
                if shortened_prompt is not None:
                    shortened_prompt = shortened_prompt[:self.max_log_len]
                if shortened_token_ids is not None:
                    shortened_token_ids = shortened_token_ids[:self.
                                                              max_log_len]
            logger.info(f"Received request {request_id}: "
                        f"prompt: {shortened_prompt!r}, "
                        f"sampling params: {sampling_params}, "
                        f"prompt token ids: {shortened_token_ids}.")

        if not self.is_running:
            if self.start_engine_loop:
                self.start_background_loop()
            else:
                raise AsyncEngineDeadError(
                    "Background loop is not running. If it was running, "
                    "inspect the output to find the stacktrace of the "
                    "error that caused the background loop to stop "
                    "(AsyncEngineDeadError).")

        stream = self._request_tracker.add_request(
            request_id,
            prompt=prompt,
            sampling_params=sampling_params,
            prompt_token_ids=prompt_token_ids,
            arrival_time=arrival_time,
            prompt_embeds=prompt_embeds,
        )

        return stream

    async def generate(
        self,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        request_id: str,
        prompt_token_ids: Optional[List[int]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
    ) -> RequestOutput:
        """Generate outputs for a request.

        Generate outputs for a request. This method is a coroutine. It adds the
        request into the waiting queue of the LLMEngine and streams the outputs
        from the LLMEngine to the caller.

        Args:
            prompt: The prompt string. Can be None if prompt_token_ids is
                provided.
            sampling_params: The sampling parameters of the request.
            request_id: The unique id of the request.
            prompt_token_ids: The token IDs of the prompt. If None, we
                use the tokenizer to convert the prompts to token IDs.

        Yields:
            The output `RequestOutput` objects from the LLMEngine for the
            request.
        """
        # Preprocess the request.
        # This should not be used for logging, as it is monotonic time.
        arrival_time = time.monotonic()

        try:
            stream = await self.add_request(
                request_id,
                prompt,
                sampling_params,
                prompt_token_ids=prompt_token_ids,
                arrival_time=arrival_time,
                prompt_embeds=prompt_embeds,
            )

            async for request_output in stream:
                yield request_output
        except (Exception, asyncio.CancelledError) as e:
            # If there is an exception or coroutine is cancelled, abort the
            # request.
            self._abort(request_id)
            raise e

    async def abort(self, request_id: str) -> None:
        """Abort a request.

        Abort a submitted request. If the request is finished or not found,
        this method will be a no-op.

        Args:
            request_id: The unique id of the request.
        """
        if not self.is_running:
            raise AsyncEngineDeadError(
                "Background loop is not running. If it was running, "
                "inspect the output to find the stacktrace of the "
                "error that caused the background loop to stop "
                "(AsyncEngineDeadError).")

        return self._abort(request_id)

    def _abort(self, request_id: str) -> None:
        """Abort a request.

        Abort a submitted request. If the request is finished or not found,
        this method will be a no-op.

        Args:
            request_id: The unique id of the request.
        """
        self._request_tracker.abort_request(request_id,
                                            verbose=self.log_requests)

    async def get_model_config(self) -> ModelConfig:
        """Get the model configuration of the vLLM engine."""
        if self.engine_use_ray:
            return await self.engine.get_model_config.remote()
        else:
            return self.engine.get_model_config()

    @classmethod
    def from_engine_args(cls,
                         engine_args: AsyncEngineArgs,
                         start_engine_loop: bool = True) -> "AsyncLLMEngine":
        """Creates an async LLM engine from the engine arguments."""
        # Create the engine configs.
        engine_configs = engine_args.create_engine_configs()
        parallel_config = engine_configs[2]
        # Initialize the cluster.
        distributed_init_method, placement_group = initialize_cluster(
            parallel_config, engine_args.engine_use_ray)
        # Create the async LLM engine.
        engine = cls(parallel_config.worker_use_ray,
                     engine_args.engine_use_ray,
                     *engine_configs,
                     distributed_init_method,
                     placement_group,
                     log_requests=not engine_args.disable_log_requests,
                     log_stats=not engine_args.disable_log_stats,
                     max_log_len=engine_args.max_log_len,
                     start_engine_loop=start_engine_loop)
        return engine


class EncoderRequest:
    def __init__(self, request_id: str,
                 multimodal_inputs: List[Tuple[MMInputType, Any]]):
        self.request_id = request_id
        self.multimodal_inputs = multimodal_inputs
        self.event = asyncio.Event()
        self.encoded_result = [None for _ in range(len(multimodal_inputs))]
        self.n_inputs_encoded = False

    async def wait(self):
        await self.event.wait()

class AsyncMLLMEngine(AsyncLLMEngine):
    """An wrapper around AsyncLLMEngine for multi-modal models.

    This class is used to wrap the AsyncLLMEngine class, so additional
    encoders can be executed to embed the multi-modal inputs before
    sending them to the AsyncLLMEngine for decoding. The encoder
    is executed in an asynchronous stream.

    NOTE: For the comprehensive list of arguments, see `LLMEngine`.

    Args:
        start_encoders_loop: If True, the background task to run the encoders
            will be automatically started in the generate call.
        *args, *kwargs: Arguments for AsyncLLMEngine.
    """
    def __init__(self,
                 *args,
                 encoder_max_batch_size: int = 32,
                 encoder_max_pending_requests: int = 16,
                 start_encoders_loop: bool = True,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        model_config = self.engine.get_model_config()
        # TODO: currently assume only one encoder
        (encoder,
         tokenize_and_postprocess_fn,
         preprocess_and_collate_fn,
         prompt_mixer) = get_multimodal_encoder(model_config)

        self.modality_encoders = {MMInputType.IMAGE: encoder}
        self.modality_encoders = {k: v.to("cuda") 
                                  for k, v in self.modality_encoders.items()}
        self.encoder_stream = torch.cuda.Stream()
        self.tokenize_and_postprocess_fn = tokenize_and_postprocess_fn
        self.preprocess_and_collate_fn = preprocess_and_collate_fn
        self.prompt_mixer = prompt_mixer

        self.encoder_max_batch_size = encoder_max_batch_size
        self.encoder_max_pending_requests = encoder_max_pending_requests

        self.tokenizer = self.engine.tokenizer
        self.encoders_background_loop = None
        # We need to keep a reference to unshielded
        # task as well to prevent it from being garbage
        # collected
        self._encoders_background_loop_unshielded = None
        self.start_encoders_loop = start_encoders_loop
        self._encoders_requests = None
        self._encoded_and_unsubmitted_requests = 0

    @property
    def encoders_are_running(self) -> bool:
        return (self.encoders_background_loop is not None
                and not self.encoders_background_loop.done())

    def start_encoders_background_loop(self) -> None:
        """Start the encoders background loop."""
        if self.encoders_are_running:
            raise RuntimeError("Encoders background loop is already running.")

        self._encoders_requests = asyncio.Queue()
        self._encoders_background_loop_unshielded = asyncio.get_event_loop(
        ).create_task(self.run_encoders_loop())
        self._encoders_background_loop_unshielded.add_done_callback(
            _raise_exception_on_finish_encoders)
        self.encoders_background_loop = asyncio.shield(
                                    self._encoders_background_loop_unshielded)

    async def run_encoders_loop(self):
        # Initialize the RequestTracker here so it uses the right event loop.
        while True:
            reqs: List[EncoderRequest] = []
            # check vLLM engine's block usage
            if (len(self.engine.scheduler.waiting) 
                    + self._encoded_and_unsubmitted_requests
                    >= self.encoder_max_pending_requests):
                # wait for requests to finish
                await asyncio.sleep(0.1)
                continue
            req = await self._encoders_requests.get()
            reqs.append(req)
            # get more requests if there are any
            while not self._encoders_requests.empty():
                if len(reqs) >= self.encoder_max_batch_size:
                    break
                req = self._encoders_requests.get_nowait()
                reqs.append(req)
            self._encoded_and_unsubmitted_requests += len(reqs)
            await self.encoders_step(reqs)
            # IMPORTANT: yield control to vLLM engine
            await asyncio.sleep(0)

    @torch.inference_mode()
    async def encoders_step(self, reqs: List[EncoderRequest]) -> None:
        """Run encoders to process the requests."""
        # sort the requests by modality
        reqs_by_modality = {}
        for req in reqs:
            for (
                input_idx, (modality, req_input)
            ) in enumerate(req.multimodal_inputs):
                if modality not in reqs_by_modality:
                    reqs_by_modality[modality] = {"reqs": [], "inputs": []}
                reqs_by_modality[modality]["reqs"].append((req, input_idx))
                reqs_by_modality[modality]["inputs"].append(req_input)
        batched_reqs = {}
        for modality, requests in reqs_by_modality.items():
            batched_reqs[modality] = self.preprocess_and_collate_fn(
                                        requests["inputs"])
        # run encoders
        with torch.cuda.stream(self.encoder_stream):
            for modality, req_inputs in batched_reqs.items():
                encoded = self.modality_encoders[modality](req_inputs)
                # we assume that the encoder returns a single tensor
                # whose first dimension is the batch size
                for (
                    batch_idx, (req, input_idx)
                    ) in enumerate(reqs_by_modality[modality]["reqs"]):
                    sliced_features = encoded[batch_idx]
                    req.encoded_result[input_idx] = (modality, sliced_features)
        # sync the stream
        self.encoder_stream.synchronize()
        for req in reqs:
            req.event.set()

    async def add_request(
        self,
        request_id: str,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[List[int]] = None,
        arrival_time: Optional[float] = None,
        modality_inputs: Optional[List[Tuple[MMInputType, Any]]] = None,
    ) -> AsyncStream:
        if self.log_requests:
            shortened_prompt = prompt
            shortened_token_ids = prompt_token_ids
            if self.max_log_len is not None:
                if shortened_prompt is not None:
                    shortened_prompt = shortened_prompt[:self.max_log_len]
                if shortened_token_ids is not None:
                    shortened_token_ids = shortened_token_ids[:self.
                                                              max_log_len]
            logger.info(f"Received request {request_id}: "
                        f"prompt: {shortened_prompt!r}, "
                        f"sampling params: {sampling_params}, "
                        f"prompt token ids: {shortened_token_ids}.")

        if not self.is_running:
            if self.start_engine_loop:
                self.start_background_loop()
            else:
                raise AsyncEngineDeadError(
                    "Background loop is not running. If it was running, "
                    "inspect the output to find the stacktrace of the "
                    "error that caused the background loop to stop "
                    "(AsyncEngineDeadError).")
        if not self.encoders_are_running:
            if self.start_encoders_loop:
                self.start_encoders_background_loop()
            else:
                raise AsyncEngineDeadError(
                    "Encoders background loop is not running. "
                    "If it was running, inspect the output to find the "
                    "stacktrace of the error that caused the background loop "
                    "to stop (AsyncEngineDeadError).")

        if modality_inputs is not None:
            # add the request to the encoder queue
            req = EncoderRequest(request_id, modality_inputs)
            self._encoders_requests.put_nowait(req)
            # tokenize the prompt if necessary
            if prompt_token_ids is None:
                prompt_token_ids = self.tokenize_and_postprocess_fn(prompt,
                                                               self.tokenizer)
            # wait for the encoder to finish
            await req.wait()
            # get the prompt embeds
            prompt_ids_for_emb = [max(0, x) for x in prompt_token_ids]
            prompt_embeds = await self.engine.embed_inputs_async(prompt_ids_for_emb)
            prompt_embeds = self.prompt_mixer(prompt_token_ids, prompt_embeds, req.encoded_result)
        else:
            prompt_embeds = None
        self._encoded_and_unsubmitted_requests -= 1
        stream = self._request_tracker.add_request(
            request_id,
            prompt=prompt,
            sampling_params=sampling_params,
            prompt_token_ids=prompt_token_ids,
            arrival_time=arrival_time,
            prompt_embeds=prompt_embeds,
        )

        return stream

    @classmethod
    def from_engine_args(cls,
                        engine_args: AsyncEngineArgs,
                        start_engine_loop: bool = True,
                        start_encoders_loop: bool = True,
                        encoder_max_batch_size: int = 32,
                        encoder_max_pending_requests: int = 16,
                        ) -> "AsyncMLLMEngine":
        """Creates an async MLLM engine from the engine arguments."""
        # Create the engine configs.
        engine_configs = engine_args.create_engine_configs()
        parallel_config = engine_configs[2]
        # Initialize the cluster.
        distributed_init_method, placement_group = initialize_cluster(
            parallel_config, engine_args.engine_use_ray)
        # Create the async LLM engine.
        engine = cls(parallel_config.worker_use_ray,
                     engine_args.engine_use_ray,
                     *engine_configs,
                     distributed_init_method,
                     placement_group,
                     log_requests=not engine_args.disable_log_requests,
                     log_stats=not engine_args.disable_log_stats,
                     max_log_len=engine_args.max_log_len,
                     start_engine_loop=start_engine_loop,
                     start_encoders_loop=start_encoders_loop,
                     encoder_max_batch_size=encoder_max_batch_size,
                     encoder_max_pending_requests=encoder_max_pending_requests)
        return engine