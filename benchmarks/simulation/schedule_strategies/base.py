from typing import Dict, List, Tuple, Optional

from ..dependency_graph import GraphNode
from vllm.transformers_utils.cost_model import ModelComponent

# strategy registry
_SCHEDULE_STRATEGIES = {}

def register_schedule_strategy(name: str):
    def decorator(cls):
        _SCHEDULE_STRATEGIES[name] = cls
        return cls
    return decorator

def get_available_schedule_strategies():
    return _SCHEDULE_STRATEGIES.keys()

def get_schedule_strategy(name: str, *args, **kwargs):
    return _SCHEDULE_STRATEGIES[name](*args, **kwargs)

class RequestStats:
    def __init__(self,
                 req_id: int,
                 enqueue_time: Optional[float] = 0.0,
                 ) -> None:
        self.req_id: int = req_id
        self.enqueue_time: float = enqueue_time
        self._per_token_finish_time: List[float] = []

    def record_token_finish(self, token_finish_time: float) -> None:
        self._per_token_finish_time.append(token_finish_time)

    def first_token_latency(self) -> float:
        return self._per_token_finish_time[0] - self.enqueue_time

    def avg_latency(self, include_first_token: bool = False) -> float:
        per_token_time = [self._per_token_finish_time[i] - self._per_token_finish_time[i-1]
                            for i in range(2, len(self._per_token_finish_time))]
        if include_first_token:
            per_token_time.append(self.first_token_latency())
        return sum(per_token_time) / len(per_token_time)

    def get_per_token_latencies(self) -> List[float]:
        return [self._per_token_finish_time[0] - self.enqueue_time] + \
               [self._per_token_finish_time[i] - self._per_token_finish_time[i-1]
                        for i in range(2, len(self._per_token_finish_time))]

    def time_since_last_token(self, current_time: float) -> float:
        return current_time - self._per_token_finish_time[-1]

class ScheduleStrategy:
    def schedule(self,
                 stats_dict: Dict[int, RequestStats],
                 ready_nodes: List[GraphNode],
                 max_batch_size: Optional[int] = None,
                 ) -> Tuple[ModelComponent, List[GraphNode]]:
        raise NotImplementedError()
