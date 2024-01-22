from typing import Dict, List, Optional, Tuple
from .base import ScheduleStrategy, register_schedule_strategy
from ..simulator import RequestStats
from ..dependency_graph import GraphNode, AttnNode, ExpertNode
from ..logger import logger
from vllm.transformers_utils.cost_model import ModelComponent

@register_schedule_strategy('FCFS')
class FCFS(ScheduleStrategy):
    def __init__(self,
                 n_layers: int,
                 per_token_latency_slo_ms: float
                 ) -> None:
        self.n_layers: int = n_layers
        self.per_token_latency_slo_ms: float = per_token_latency_slo_ms
        self._current_batch_requests: List[Tuple(int, int)] = []
        self._prev_batch_requests: List[Tuple(int, int)] = []
        self._current_layer: int = 0
        self._current_phase: type[GraphNode] = AttnNode
        self._activated_experts_per_batch = 0
        self._activated_experts_history = []

    def _advance_phase(self) -> None:
        if self._current_phase == AttnNode:
            self._current_phase = ExpertNode
        else:
            self._current_phase = AttnNode
            self._current_layer += 1
        logger.debug("Advanced phase to {}, layer {}.".format(
            self._current_phase.__name__, 
            self._current_layer)
        )

    def _reset_batch(self) -> None:
        self._prev_batch_requests = self._current_batch_requests
        self._current_batch_requests = []
        self._current_layer = 0
        self._current_phase = AttnNode
        self._activated_experts_history.append(self._activated_experts_per_batch)
        self._activated_experts_per_batch = 0

    def schedule(self,
                 stats_dict: Dict[int, RequestStats],
                 ready_nodes: List[GraphNode],
                 max_batch_size: Optional[int] = None,
                 ) -> Tuple[ModelComponent, List[GraphNode]]:
        if not ready_nodes:
            # no more requests to schedule
            return None, None
        # execute the same requests layer by layer
        if not self._current_batch_requests:
            assert self._current_layer == 0
            # first try to execute the next token of the current batch
            ready_first_layer_attn_nodes = []
            for node in ready_nodes:
                if (node.req_id, node.token_index - 1) in self._prev_batch_requests:
                    ready_first_layer_attn_nodes.append(node)
                    self._current_batch_requests.append((node.req_id, node.token_index))
            if len(ready_first_layer_attn_nodes) < max_batch_size:
                # schedule all first layer Attn requests in FCFS order
                for node in sorted(ready_nodes, key=lambda node: stats_dict[node.req_id].enqueue_time):
                    if node.layer_id == self._current_layer and isinstance(node, AttnNode):
                        if (node.req_id, node.token_index) not in self._current_batch_requests:
                            ready_first_layer_attn_nodes.append(node)
                            self._current_batch_requests.append((node.req_id, node.token_index))
                            if len(ready_first_layer_attn_nodes) == max_batch_size:
                                break
            # truncate to max batch size
            logger.debug("Scheduling first batch of {} requests.".format(len(self._current_batch_requests)))
            assert len(self._current_batch_requests) > 0
        # first test if the current batch is all finished
        # we assume all nodes are DecodeNodes
        current_batch_ready_nodes = [node for node in ready_nodes
                                        if (node.req_id, node.token_index) in self._current_batch_requests]
        if not current_batch_ready_nodes:
            # # DEBUG
            # logger.debug("Current batch finished, reset batch")
            # return None, None
            self._reset_batch()
            return self.schedule(stats_dict, ready_nodes, max_batch_size)
        # execute the current batch
        current_batch_ready_nodes = [node for node in current_batch_ready_nodes if
                                     isinstance(node, self._current_phase)
                                     and node.layer_id == self._current_layer]
        if not current_batch_ready_nodes:
            # advance to the next phase
            self._advance_phase()
            current_batch_ready_nodes = [node for node in ready_nodes if
                                     isinstance(node, self._current_phase)
                                     and node.layer_id == self._current_layer]
        if self._current_phase == ExpertNode:
            # schedule one expert at a time
            current_batch_ready_nodes: List[ExpertNode]
            current_batch_ready_nodes = [node for node in current_batch_ready_nodes
                                         if node.expert_id == current_batch_ready_nodes[0].expert_id]
            logger.debug("Scheduling {} expert nodes: Layer {}, Expert: {}".format(
                len(current_batch_ready_nodes),
                self._current_layer,
                current_batch_ready_nodes[0].expert_id
            ))
            self._activated_experts_per_batch += 1
            return ModelComponent.EXPERT, current_batch_ready_nodes
        else:
            # schedule all attn nodes together
            logger.debug("Scheduling {} attn nodes: Layer {}".format(len(current_batch_ready_nodes), self._current_layer))
            return ModelComponent.ATTENTION_GATE, current_batch_ready_nodes

